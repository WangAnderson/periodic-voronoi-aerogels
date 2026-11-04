"""Export helpers for COMSOL, Lumerical, and quick visual inspection.

The modeling code uses nanometres internally.  STL files deliberately keep
those numeric coordinates (STL has no unit field), while NASTRAN BDF files are
written in micrometres to keep fixed-width numeric fields well conditioned.

本模块把周期逻辑面裁剪到单个基本胞元，生成 COMSOL 可导入的 NASTRAN 壳网格
及其周期节点配对；同时导出 Lumerical 所需的实体 STL 导入脚本、预览图和说明
文件。内部计算仍使用 nm，BDF 为改善定宽字段数值表示而改用 µm。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import json
import math
import os
import tempfile

import numpy as np

from .mesh import TriangleMesh, face_normal, ordered_face_vertices


def _value(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _select_faces(faces: Sequence[object], selected_ids: Sequence[int] | None) -> list[tuple[int, object]]:
    face_list = list(faces)
    wanted = None if selected_ids is None else {int(item) for item in selected_ids}
    result: list[tuple[int, object]] = []
    for index, face in enumerate(face_list):
        face_id = int(_value(face, "id", index))
        if wanted is None or face_id in wanted:
            result.append((face_id, face))
    return result


def _clip_halfspace(
    polygon: np.ndarray,
    axis: int,
    bound: float,
    keep_greater: bool,
    tolerance: float,
) -> np.ndarray:
    """用一个轴对齐半空间裁剪多边形（Sutherland--Hodgman 步骤）。"""

    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[axis] >= bound - tolerance)
        return bool(point[axis] <= bound + tolerance)

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(float(denominator)) > np.finfo(float).eps:
                fraction = (bound - previous[axis]) / denominator
                point = previous + fraction * (current - previous)
                point[axis] = bound
                output.append(point)
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 3), dtype=float)
    return np.asarray(output, dtype=float)


def _deduplicate_polygon_vertices(polygon: np.ndarray, tolerance: float) -> np.ndarray:
    if len(polygon) < 2:
        return polygon
    cleaned: list[np.ndarray] = []
    for point in polygon:
        if not cleaned or np.linalg.norm(point - cleaned[-1], ord=np.inf) > tolerance:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1], ord=np.inf) <= tolerance:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def _polygon_area(polygon: np.ndarray, normal: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(
                np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0),
                normal,
            )
        )
    )


def clip_periodic_face_patches(
    face: object,
    cube_size_nm: float,
    *,
    tolerance_nm: float | None = None,
) -> list[np.ndarray]:
    """Return all translated pieces of a convex periodic face inside a cube.

    一个周期逻辑面可能跨越基本胞元的一至多个边界。本函数枚举其必要的平移
    镜像，逐一用六个盒半空间裁剪，并返回全部非退化片段；所有片段仍属于同一
    逻辑面，因此保留/删除操作不会破坏周期一致性。
    """

    box_size = float(cube_size_nm)
    if box_size <= 0.0:
        raise ValueError("cube_size_nm must be positive")
    tolerance = (
        max(box_size * 1e-10, 1e-9)
        if tolerance_nm is None
        else float(tolerance_nm)
    )
    if tolerance <= 0.0:
        raise ValueError("tolerance_nm must be positive")
    polygon, normal = ordered_face_vertices(face)
    lower = polygon.min(axis=0)
    upper = polygon.max(axis=0)
    shift_ranges: list[range] = []
    for lo, hi in zip(lower, upper, strict=True):
        first = int(math.ceil((-hi - tolerance) / box_size))
        last = int(math.floor((box_size - lo + tolerance) / box_size))
        shift_ranges.append(range(first, last + 1))

    patches: list[np.ndarray] = []
    signatures: set[tuple[int, ...]] = set()
    # 对可能与 [0,L]^3 相交的周期镜像逐个执行六平面裁剪。
    for shift_index in product(*shift_ranges):
        patch = polygon + box_size * np.asarray(shift_index, dtype=float)
        for axis in range(3):
            patch = _clip_halfspace(patch, axis, 0.0, True, tolerance)
            patch = _clip_halfspace(patch, axis, box_size, False, tolerance)
            if len(patch) < 3:
                break
        if len(patch) < 3:
            continue
        patch = _deduplicate_polygon_vertices(patch, tolerance)
        if len(patch) < 3 or _polygon_area(patch, normal) <= tolerance * tolerance:
            continue
        patch[np.abs(patch) <= tolerance] = 0.0
        patch[np.abs(patch - box_size) <= tolerance] = box_size
        # 裁剪通常保持原始环绕方向，但数值重复点可能使方向含糊；这里依据源面
        # 法向重新排序，确保后续三角形法向一致。
        patch, _ = ordered_face_vertices(patch, patch, normal)
        quantized = np.rint(patch / tolerance).astype(np.int64)
        ordered_rows = quantized[np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))]
        signature = tuple(int(item) for item in ordered_rows.ravel())
        # 量化后的无序顶点集作为签名，去除恰落在周期边界时产生的重复片段。
        if signature in signatures:
            continue
        signatures.add(signature)
        patches.append(patch)
    return patches


def triangulate_shell_faces(
    faces: Sequence[object],
    selected_ids: Sequence[int] | None,
    cube_size_nm: float,
    *,
    merge_tolerance_nm: float | None = None,
) -> dict[str, np.ndarray]:
    """Clip periodic faces to the basic cell and fan-triangulate the patches.

    先把周期面裁剪到基本胞元，再以扇形方式三角化。相交片层及相邻裁剪片段在
    容差内共享同一个 GRID 节点，这对 COMSOL 壳模型中的温度连续性至关重要。
    """

    box_size = float(cube_size_nm)
    tolerance = (
        max(box_size * 1e-9, 1e-7)
        if merge_tolerance_nm is None
        else float(merge_tolerance_nm)
    )
    if tolerance <= 0.0:
        raise ValueError("merge_tolerance_nm must be positive")

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_source_ids: list[int] = []
    vertex_lookup: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def vertex_index(point: np.ndarray) -> int:
        # 空间哈希同时搜索相邻量化桶，避免位于桶边界两侧的同一点未被合并。
        key = tuple(int(item) for item in np.rint(point / tolerance))
        for offset in product((-1, 0, 1), repeat=3):
            neighbour = tuple(key[axis] + offset[axis] for axis in range(3))
            for existing in vertex_lookup.get(neighbour, ()):
                if np.linalg.norm(point - vertices[existing], ord=np.inf) <= tolerance:
                    return existing
        index = len(vertices)
        vertices.append(point.copy())
        vertex_lookup[key].append(index)
        return index

    for face_id, face in _select_faces(faces, selected_ids):
        normal = face_normal(face)
        for patch in clip_periodic_face_patches(
            face,
            box_size,
            tolerance_nm=min(tolerance, max(box_size * 1e-10, 1e-9)),
        ):
            indices = [vertex_index(point) for point in patch]
            for offset in range(1, len(indices) - 1):
                triangle = (indices[0], indices[offset], indices[offset + 1])
                points = np.asarray([vertices[index] for index in triangle])
                area_vector = np.cross(points[1] - points[0], points[2] - points[0])
                if np.linalg.norm(area_vector) <= tolerance * tolerance:
                    continue
                # 统一三角形绕序，使其法向与原逻辑面的法向相同。
                if float(np.dot(area_vector, normal)) < 0.0:
                    triangle = (triangle[0], triangle[2], triangle[1])
                triangles.append(triangle)
                triangle_source_ids.append(face_id)

    vertex_array = np.asarray(vertices, dtype=float).reshape((-1, 3))
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    source_array = np.asarray(triangle_source_ids, dtype=np.int64)
    return {
        "vertices_nm": vertex_array,
        "triangles": triangle_array,
        "source_face_ids": source_array,
    }


def _nastran_float8(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("NASTRAN fields cannot contain NaN or infinite values")
    if value == 0.0:
        return "0."
    for significant in range(7, 0, -1):
        text = f"{value:.{significant}g}".upper()
        if len(text) <= 8:
            return text
    raise ValueError(f"value {value!r} does not fit an 8-character NASTRAN field")


def _field8(value: object) -> str:
    if value is None or value == "":
        return " " * 8
    if isinstance(value, (float, np.floating)):
        text = _nastran_float8(float(value))
    else:
        text = str(value)
    if len(text) > 8:
        raise ValueError(f"NASTRAN field {text!r} exceeds 8 characters")
    return text.rjust(8)


def _bdf_card(*fields: object) -> str:
    if not fields:
        return "\n"
    card_name = str(fields[0])
    if len(card_name) > 8:
        raise ValueError(f"NASTRAN card name {card_name!r} exceeds 8 characters")
    return card_name.ljust(8) + "".join(_field8(value) for value in fields[1:]).rstrip() + "\n"


def export_shell_bdf(
    faces: Sequence[object],
    selected_ids: Sequence[int] | None,
    cube_size_nm: float,
    thickness_nm: float,
    path: str | Path,
    *,
    property_id: int = 1,
    material_id: int = 1,
    merge_tolerance_nm: float | None = None,
    periodic_pairs_path: str | Path | None = None,
) -> dict[str, object]:
    """Write a fixed-field NASTRAN shell mesh in micrometre units.

    ``MAT1`` is only a structural placeholder needed by ``PSHELL``; thermal
    conductivity and heat capacity are intentionally assigned in COMSOL.

    BDF 中 GRID 坐标和 PSHELL 厚度均使用 µm。MAT1 只为满足 PSHELL 引用关系，
    不代表氧化铝热物性；导入 COMSOL 后必须由用户另行设置导热系数、密度和比热。
    """

    if thickness_nm <= 0.0:
        raise ValueError("thickness_nm must be positive")
    shell = triangulate_shell_faces(
        faces,
        selected_ids,
        cube_size_nm,
        merge_tolerance_nm=merge_tolerance_nm,
    )
    vertices_nm = shell["vertices_nm"]
    triangles = shell["triangles"]
    if len(vertices_nm) > 99_999_999 or len(triangles) > 99_999_999:
        raise ValueError("small-field BDF IDs are limited to eight digits")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 小字段 BDF 每栏最多八字符；用 µm 可减少 1e-3 量级坐标的表示损失。
    lines = [
        "$ Periodic Voronoi aerogel shell mesh\n",
        "$ LENGTH UNIT: micrometre (um); internal generator unit: nm\n",
        "$ MAT1 below is a structural placeholder, not a thermal material.\n",
        "$ Assign k, rho, and Cp in COMSOL Heat Transfer in Shells.\n",
        "CEND\n",
        "BEGIN BULK\n",
        _bdf_card("MAT1", material_id, 1.0, None, 0.3),
        _bdf_card("PSHELL", property_id, material_id, float(thickness_nm) / 1000.0),
    ]
    for node_id, coordinate_nm in enumerate(vertices_nm, start=1):
        x, y, z = coordinate_nm / 1000.0
        lines.append(_bdf_card("GRID", node_id, 0, float(x), float(y), float(z)))
    for element_id, triangle in enumerate(triangles, start=1):
        lines.append(
            _bdf_card(
                "CTRIA3",
                element_id,
                property_id,
                int(triangle[0]) + 1,
                int(triangle[1]) + 1,
                int(triangle[2]) + 1,
            )
        )
    lines.append("ENDDATA\n")
    target.write_text("".join(lines), encoding="ascii", newline="\n")

    # BDF 只携带网格拓扑，周期热边界需由配套 JSON 在 COMSOL 中另行建立。
    pair_data = None
    if periodic_pairs_path is not None:
        pair_data = write_periodic_pairs_json(
            periodic_pairs_path,
            vertices_nm,
            cube_size_nm,
            tolerance_nm=merge_tolerance_nm,
        )
    return {
        "path": target,
        "vertices_nm": vertices_nm,
        "triangles": triangles,
        "source_face_ids": shell["source_face_ids"],
        "node_ids": np.arange(1, len(vertices_nm) + 1, dtype=np.int64),
        "element_ids": np.arange(1, len(triangles) + 1, dtype=np.int64),
        "length_unit": "um",
        "thickness_um": float(thickness_nm) / 1000.0,
        "periodic_pairs": pair_data,
    }


def _boundary_pairs_for_axis(
    vertices: np.ndarray,
    node_ids: np.ndarray,
    axis: int,
    box_size: float,
    tolerance: float,
) -> tuple[list[dict[str, object]], list[int], list[int]]:
    """按横向坐标匹配某一轴负、正边界上的壳网格节点。"""

    transverse = [value for value in range(3) if value != axis]
    minus_index = np.flatnonzero(np.abs(vertices[:, axis]) <= tolerance)
    plus_index = np.flatnonzero(np.abs(vertices[:, axis] - box_size) <= tolerance)

    # 将正边界节点按两个横向坐标散列，匹配时只需搜索相邻九个量化桶。
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in plus_index:
        key = tuple(int(value) for value in np.rint(vertices[index, transverse] / tolerance))
        buckets[key].append(int(index))

    used: set[int] = set()
    pairs: list[dict[str, object]] = []
    unmatched_minus: list[int] = []
    for source_index in minus_index:
        base_key = np.rint(vertices[source_index, transverse] / tolerance).astype(np.int64)
        candidates: list[int] = []
        for delta_a in (-1, 0, 1):
            for delta_b in (-1, 0, 1):
                candidates.extend(
                    buckets.get((int(base_key[0] + delta_a), int(base_key[1] + delta_b)), [])
                )
        candidates = [index for index in candidates if index not in used]
        if candidates:
            distances = np.max(
                np.abs(vertices[candidates][:, transverse] - vertices[source_index, transverse]),
                axis=1,
            )
            best_position = int(np.argmin(distances))
            target_index = candidates[best_position]
            if float(distances[best_position]) <= tolerance:
                used.add(target_index)
                pairs.append(
                    {
                        "minus_node_id": int(node_ids[source_index]),
                        "plus_node_id": int(node_ids[target_index]),
                        "minus_coordinate_nm": vertices[source_index].tolist(),
                        "plus_coordinate_nm": vertices[target_index].tolist(),
                    }
                )
                continue
        unmatched_minus.append(int(node_ids[source_index]))
    unmatched_plus = [int(node_ids[index]) for index in plus_index if int(index) not in used]
    pairs.sort(key=lambda item: int(item["minus_node_id"]))
    return pairs, unmatched_minus, unmatched_plus


def _equivalence_classes(node_ids: np.ndarray, axis_pairs: Mapping[str, object]) -> list[list[int]]:
    """用并查集合并棱边/角点上经多个方向周期配对后等价的节点。"""

    parent = {int(node_id): int(node_id) for node_id in node_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for data in axis_pairs.values():
        for pair in data["pairs"]:
            union(int(pair["minus_node_id"]), int(pair["plus_node_id"]))
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id in node_ids:
        groups[find(int(node_id))].append(int(node_id))
    return sorted((sorted(group) for group in groups.values() if len(group) > 1), key=lambda x: x[0])


def write_periodic_pairs_json(
    path: str | Path,
    vertices_nm: np.ndarray | Mapping[str, object],
    cube_size_nm: float,
    *,
    tolerance_nm: float | None = None,
    node_ids: Sequence[int] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write opposite-boundary node matches for a shell BDF.

    This sidecar is descriptive: NASTRAN does not encode a COMSOL heat-periodic
    pair, so a COMSOL model or script must consume these matches separately.

    JSON 同时记录 x/y/z 三组负、正边界节点、平移向量、未匹配节点及角点/棱边
    等价类。它是建立 COMSOL 周期约束的辅助数据，并不会自动激活热周期条件。
    """

    if isinstance(vertices_nm, Mapping):
        values = vertices_nm.get("vertices_nm", vertices_nm.get("vertices"))
        if values is None:
            raise ValueError("mesh mapping needs vertices_nm")
        vertices = np.asarray(values, dtype=float)
        if node_ids is None and "node_ids" in vertices_nm:
            node_ids = vertices_nm["node_ids"]
    else:
        vertices = np.asarray(vertices_nm, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_nm must have shape (N, 3)")
    box_size = float(cube_size_nm)
    tolerance = (
        max(box_size * 1e-8, 1e-6)
        if tolerance_nm is None
        else float(tolerance_nm)
    )
    if tolerance <= 0.0:
        raise ValueError("tolerance_nm must be positive")
    ids = (
        np.arange(1, len(vertices) + 1, dtype=np.int64)
        if node_ids is None
        else np.asarray(node_ids, dtype=np.int64)
    )
    if ids.shape != (len(vertices),) or len(np.unique(ids)) != len(ids):
        raise ValueError("node_ids must be unique and match the vertex count")

    # 三个方向分别配对，平移向量始终由负边界指向正边界。
    axes: dict[str, object] = {}
    axis_names = ("x", "y", "z")
    for axis, name in enumerate(axis_names):
        pairs, unmatched_minus, unmatched_plus = _boundary_pairs_for_axis(
            vertices, ids, axis, box_size, tolerance
        )
        translation = [0.0, 0.0, 0.0]
        translation[axis] = box_size
        axes[name] = {
            "minus_plane_nm": 0.0,
            "plus_plane_nm": box_size,
            "translation_nm": translation,
            "pairs": pairs,
            "unmatched_minus_node_ids": unmatched_minus,
            "unmatched_plus_node_ids": unmatched_plus,
        }

    data: dict[str, object] = {
        "schema": "voronoi-aerogel-periodic-node-pairs-v1",
        "coordinate_unit": "nm",
        "bdf_node_coordinate_unit": "um",
        "cube_bounds_nm": [[0.0, box_size], [0.0, box_size], [0.0, box_size]],
        "matching_tolerance_nm": tolerance,
        "axes": axes,
        "equivalence_classes": _equivalence_classes(ids, axes),
        "note": (
            "BDF does not encode COMSOL thermal periodic constraints. "
            "Use these matches or coordinate selections in the COMSOL model."
        ),
    }
    if extra_metadata:
        data["metadata"] = dict(extra_metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


def _mesh_arrays(
    geometry: object,
    triangles: np.ndarray | None,
    selected_ids: Sequence[int] | None,
    cube_size_nm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, TriangleMesh):
        return geometry.vertices_nm, geometry.faces
    if isinstance(geometry, Mapping):
        vertices = geometry.get("vertices_nm", geometry.get("vertices"))
        connectivity = geometry.get("triangles", geometry.get("faces"))
        if vertices is not None and connectivity is not None:
            return np.asarray(vertices, dtype=float), np.asarray(connectivity, dtype=np.int64)
    if triangles is not None:
        return np.asarray(geometry, dtype=float), np.asarray(triangles, dtype=np.int64)
    if cube_size_nm is None:
        raise ValueError("cube_size_nm is required when previewing sheet polygons")
    shell = triangulate_shell_faces(geometry, selected_ids, cube_size_nm)
    return shell["vertices_nm"], shell["triangles"]


def save_preview(
    path: str | Path,
    geometry: object,
    selected_ids: Sequence[int] | None = None,
    cube_size_nm: float | None = None,
    *,
    triangles: np.ndarray | None = None,
    max_faces: int = 2_000,
    dpi: int = 180,
    title: str = "Periodic Voronoi sheet network",
) -> Path:
    """Save a dependency-light PNG preview of a shell or solid mesh.

    大网格仅均匀抽取有限数量三角形用于展示，不影响实际导出的 BDF/STL。
    """

    if max_faces <= 0:
        raise ValueError("max_faces must be positive")
    cache_dir = Path(tempfile.gettempdir()) / "voronoi_aerogel_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    vertices, connectivity = _mesh_arrays(geometry, triangles, selected_ids, cube_size_nm)
    # 只对预览降采样，避免数百万三角形拖慢绘图或耗尽显存/内存。
    if len(connectivity) > max_faces:
        indices = np.linspace(0, len(connectivity) - 1, max_faces, dtype=np.int64)
        connectivity = connectivity[indices]

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = Figure(figsize=(7.2, 6.4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    if len(connectivity):
        collection = Poly3DCollection(
            vertices[connectivity],
            facecolor="#d9e6f2",
            edgecolor="#315b7d",
            linewidth=0.15,
            alpha=0.82,
        )
        axis.add_collection3d(collection)

    if cube_size_nm is not None:
        box_size = float(cube_size_nm)
        corners = np.asarray(list(product((0.0, box_size), repeat=3)))
        for start in corners:
            for coordinate in range(3):
                if start[coordinate] != 0.0:
                    continue
                end = start.copy()
                end[coordinate] = box_size
                axis.plot(*zip(start, end, strict=True), color="#333333", linewidth=0.55)
        lower = np.zeros(3)
        upper = np.full(3, box_size)
    elif len(vertices):
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
    else:
        lower = np.zeros(3)
        upper = np.ones(3)
    centre = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.0)
    for setter, value in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim), centre, strict=True
    ):
        setter(value - span / 2.0, value + span / 2.0)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("y (nm)")
    axis.set_zlabel("z (nm)")
    axis.set_title(title)
    axis.view_init(elev=24, azim=35)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, facecolor="white")
    return target


def write_lumerical_script(
    path: str | Path,
    stl_filename: str | Path,
    cube_size_nm: float,
    *,
    object_name: str = "alumina_aerogel",
    refractive_index: float = 1.76,
    z_padding_nm: float | None = None,
    mesh_accuracy: int = 2,
) -> Path:
    """Write an FDTD import script for an STL whose numeric unit is nm.

    脚本用 ``stlimport(..., 1e-9, ...)`` 把 STL 数值从 nm 换算为 m，并建立
    x/y 周期、z 向 PML 的有限厚度样品计算域；这种设置才适合求透射/散射光谱。
    """

    box_size = float(cube_size_nm)
    if box_size <= 0.0:
        raise ValueError("cube_size_nm must be positive")
    padding = 0.25 * box_size if z_padding_nm is None else float(z_padding_nm)
    if padding <= 0.0:
        raise ValueError("z_padding_nm must be positive")
    filename = Path(stl_filename).as_posix().replace('"', '\\"')
    safe_name = object_name.replace('"', '\\"')
    # 这里生成的是可移植模板：材料色散、光源、监视器和收敛网格仍需按研究波段
    # 在 Lumerical 中设置，不能把占位折射率直接当作最终氧化铝光学模型。
    script = f'''# Generated Lumerical FDTD import script.
# The STL stores numeric coordinates in nm; stlimport scalingFactor=1e-9
# converts those coordinates to metres. Run this script in a fresh layout.
switchtolayout;

stl_file = "{filename}";
stlimport(stl_file, 1e-9, 1e-12, true);
set("name", "{safe_name}");
# Portable placeholder. Replace with a dispersive alumina material from your
# local material database before a production optical calculation.
set("material", "<Object defined dielectric>");
set("index", {float(refractive_index):.12g});

L = {box_size:.17g}e-9;
z_buffer = {padding:.17g}e-9;
addfdtd;
set("name", "FDTD_unit_cell");
set("dimension", "3D");
set("x", 0.5*L);
set("x span", L);
set("y", 0.5*L);
set("y span", L);
set("z", 0.5*L);
set("z span", L + 2*z_buffer);
set("mesh accuracy", {int(mesh_accuracy)});
set("x min bc", "Periodic");
set("x max bc", "Periodic");
set("y min bc", "Periodic");
set("y max bc", "Periodic");
set("z min bc", "PML");
set("z max bc", "PML");

# x/y use Periodic boundaries for a normally incident periodic excitation.
# For oblique incidence use Bloch boundaries and set the Bloch wavevector.
# z uses PML and a vacuum buffer so the structure never touches the PML.
# Add the source, wavelength range, monitors, and a convergence-tested mesh.
'''
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8", newline="\n")
    return target


def write_readmes(
    output_dir: str | Path,
    *,
    bdf_filename: str = "aerogel_shell.bdf",
    pairs_filename: str = "periodic_pairs.json",
    stl_filename: str = "aerogel_solid_nm.stl",
    lumerical_script_filename: str = "import_lumerical.lsf",
) -> dict[str, Path]:
    """Write concise hand-off notes for COMSOL and Lumerical users.

    说明文件明确记录单位约定、材料占位符以及周期边界的限制，防止模型虽成功
    导入，但因单位或边界条件误设而得到错误的导热或透射结果。
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comsol = directory / "README_COMSOL.md"
    lumerical = directory / "README_LUMERICAL.md"
    # COMSOL 说明强调：BDF 节点配对数据本身不是已激活的热周期约束。
    comsol.write_text(
        f"""# COMSOL shell-mesh hand-off

## Files and units

- `{bdf_filename}` is a fixed-field NASTRAN shell mesh. GRID coordinates and
  PSHELL thickness are in micrometres (µm).
- `{pairs_filename}` records opposite-boundary BDF node matches in nanometres.

Import the BDF as a mesh and verify the imported size and PSHELL thickness.
Use **Heat Transfer in Shells** for the mid-surfaces. The MAT1 card is only a
structural placeholder: assign alumina thermal conductivity, density, and heat
capacity in COMSOL.

Sheets that intersect share GRID nodes, which is required for temperature
continuity through the network. Check for unintended disconnected components.

## Periodicity limitation

NASTRAN does not encode COMSOL thermal periodic conditions. The JSON sidecar is
metadata, not an automatically active boundary condition. COMSOL 6.4 does not
document the ordinary heat-transfer Periodic Condition for the Heat Transfer in
Shells interface. A periodic shell RVE therefore needs verified custom edge
constraints (temperature jump/equality plus conservative heat flux), or an
alternative solid-domain homogenization model. Do not report effective thermal
conductivity until that constraint implementation has been benchmarked.
""",
        encoding="utf-8",
    )
    # Lumerical 说明强调 STL 无单位，以及透射问题必须保留 z 向开放/PML 边界。
    lumerical.write_text(
        f"""# Lumerical FDTD hand-off

- `{stl_filename}` is a watertight binary STL whose numeric coordinates are in
  nanometres. STL contains no unit metadata.
- Run `{lumerical_script_filename}` from the same directory. It calls
  `stlimport(..., 1e-9, ...)`, converting nm coordinates to metres.

The generated setup uses Periodic boundaries in x/y and PML in z with a buffer.
For oblique incidence replace x/y Periodic conditions with Bloch conditions and
set the appropriate wavevector. Replace the placeholder constant refractive
index with a validated dispersive alumina material, then add the source,
monitors, wavelength range, and convergence-tested local mesh. Confirm that the
solid does not touch the z PML and inspect the STL for import diagnostics.
""",
        encoding="utf-8",
    )
    return {"comsol": comsol, "lumerical": lumerical}


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
    **fields: object,
) -> Path:
    """Write JSON metadata, converting NumPy scalars and arrays safely."""

    data = {} if metadata is None else dict(metadata)
    data.update(fields)
    data.setdefault("internal_length_unit", "nm")
    data.setdefault("stl_coordinate_unit", "nm")
    data.setdefault("bdf_coordinate_unit", "um")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


write_metadata_json = write_metadata


__all__ = [
    "clip_periodic_face_patches",
    "export_shell_bdf",
    "save_preview",
    "triangulate_shell_faces",
    "write_lumerical_script",
    "write_metadata",
    "write_metadata_json",
    "write_periodic_pairs_json",
    "write_readmes",
]
