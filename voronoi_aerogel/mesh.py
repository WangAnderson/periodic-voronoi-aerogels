"""Voxel and triangle-mesh utilities for periodic sheet networks.

All geometry handled by this module is expressed in nanometres.  The
voxelizer marks membership in the *union* of the selected thick sheets;
overlaps are therefore counted only once.

本模块负责把周期 Voronoi 中面扩展为有限厚度片层，并将所有片层的物理并集
体素化；重叠区只计一次。随后通过带空白幽灵层的 marching cubes 提取封闭的
基本胞元表面，并提供水密性检查与二进制 STL 输出。内部长度单位统一为 nm。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from collections.abc import Mapping, Sequence
import math
import struct

import numpy as np
from skimage.measure import marching_cubes


@dataclass(slots=True)
class TriangleMesh:
    """A triangular surface mesh whose coordinates are in nanometres.

    三角网格只保存顶点与拓扑；顶点坐标始终以 nm 表示，避免 STL 本身没有
    单位字段所造成的歧义。
    """

    vertices_nm: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_nm, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.size == 0:
            vertices = np.empty((0, 3), dtype=np.float64)
        if faces.size == 0:
            faces = np.empty((0, 3), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_nm must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face indices are outside the vertex array")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices_nm contains NaN or infinite values")
        self.vertices_nm = vertices
        self.faces = faces

    @property
    def triangles(self) -> np.ndarray:
        """Alias useful for callers that name connectivity ``triangles``."""

        return self.faces

    @property
    def vertices(self) -> np.ndarray:
        """Conventional alias for :attr:`vertices_nm`."""

        return self.vertices_nm

    def __iter__(self):
        """Allow ``vertices, faces = mesh`` without losing unit metadata."""

        yield self.vertices_nm
        yield self.faces


def _value(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def face_vertices_nm(face: object) -> np.ndarray:
    """Return an ``(N, 3)`` polygon from an array, mapping, or face object."""

    if isinstance(face, np.ndarray):
        vertices = face
    else:
        vertices = _value(face, "vertices_nm")
        if vertices is None:
            vertices = _value(face, "vertices")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("each sheet face must contain at least three 3D vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("sheet face vertices contain NaN or infinite values")
    return vertices


def face_normal(face: object, vertices_nm: np.ndarray | None = None) -> np.ndarray:
    """Return a normalized face normal, deriving it from vertices if needed."""

    vertices = face_vertices_nm(face) if vertices_nm is None else vertices_nm
    raw = _value(face, "normal") if not isinstance(face, np.ndarray) else None
    if raw is not None:
        normal = np.asarray(raw, dtype=np.float64)
    else:
        # Newell 法对多于三个顶点的平面多边形更稳定，可避免只取前三点时
        # 因局部短边或近共线而得到不可靠的法向。
        normal = np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0)
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= np.finfo(float).eps:
        raise ValueError("sheet face has a degenerate normal")
    return normal / length


def ordered_face_vertices(
    face: object,
    vertices_nm: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Order a convex polygon around its centroid consistently with its normal."""

    vertices = face_vertices_nm(face) if vertices_nm is None else np.asarray(vertices_nm)
    n = face_normal(face, vertices) if normal is None else np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    # 先删除数值上重复的顶点，再在面内建立二维基底并按极角排序。
    # 这保证后续凸多边形内点判断及扇形三角化使用一致的环绕方向。
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    decimals = max(0, int(-math.log10(scale * 1e-12)))
    _, unique_index = np.unique(np.round(vertices, decimals=decimals), axis=0, return_index=True)
    vertices = vertices[np.sort(unique_index)]
    if len(vertices) < 3:
        raise ValueError("sheet face collapses to fewer than three unique vertices")

    centre = vertices.mean(axis=0)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, n))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, trial)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    rel = vertices - centre
    angle = np.arctan2(rel @ v, rel @ u)
    ordered = vertices[np.argsort(angle)]
    signed_area = 0.5 * float(
        np.dot(
            np.sum(np.cross(ordered, np.roll(ordered, -1, axis=0)), axis=0),
            n,
        )
    )
    if signed_area < 0.0:
        ordered = ordered[::-1]
    return ordered, n


def _selected_faces(faces: Sequence[object], selected_ids: Sequence[int] | None) -> list[object]:
    face_list = list(faces)
    if selected_ids is None:
        return face_list
    wanted = {int(value) for value in selected_ids}
    selected: list[object] = []
    for index, face in enumerate(face_list):
        face_id = _value(face, "id", index)
        if int(face_id) in wanted:
            selected.append(face)
    return selected


def effective_voxel_size_nm(cube_size_nm: float, requested_voxel_size_nm: float) -> float:
    """Return the uniform spacing used to tile a cube without a partial voxel.

    将请求步长略微缩小，使每个方向恰好由整数个体素填满基本胞元，从而避免
    周期边界两侧出现不足一个体素的残余区。
    """

    if cube_size_nm <= 0.0 or requested_voxel_size_nm <= 0.0:
        raise ValueError("cube_size_nm and requested_voxel_size_nm must be positive")
    count = max(1, int(math.ceil(cube_size_nm / requested_voxel_size_nm)))
    return float(cube_size_nm) / count


def _translation_ranges(vertices: np.ndarray, extent: np.ndarray, box_size: float) -> list[range]:
    """求出可能与基本胞元相交的全部周期镜像平移编号范围。"""

    ranges: list[range] = []
    tolerance = box_size * 1e-12
    lower = vertices.min(axis=0) - extent
    upper = vertices.max(axis=0) + extent
    for lo, hi in zip(lower, upper, strict=True):
        first = int(math.ceil((-hi - tolerance) / box_size))
        last = int(math.floor((box_size - lo + tolerance) / box_size))
        ranges.append(range(first, last + 1))
    return ranges


def _voxel_index_range(lo: float, hi: float, spacing: float, count: int) -> tuple[int, int]:
    """Inclusive centre-index interval intersecting ``[lo, hi]``."""

    first = max(0, int(math.ceil(lo / spacing - 0.5 - 1e-12)))
    last = min(count - 1, int(math.floor(hi / spacing - 0.5 + 1e-12)))
    return first, last


def _points_inside_extruded_polygon(
    points: np.ndarray,
    polygon: np.ndarray,
    normal: np.ndarray,
    half_thickness: float,
    tolerance: float,
) -> np.ndarray:
    """判断采样点是否位于由凸多边形沿法向拉伸得到的有限厚度棱柱内。"""

    centre = polygon.mean(axis=0)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, normal))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, trial)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    rel_points = points - centre
    plane_distance = rel_points @ normal
    # 先用厚度方向快速筛选，再把候选点投影到面内二维坐标系。
    near_plane = np.abs(plane_distance) <= half_thickness + tolerance
    if not np.any(near_plane):
        return near_plane

    polygon_2d = np.column_stack(((polygon - centre) @ u, (polygon - centre) @ v))
    point_2d = np.column_stack((rel_points[near_plane] @ u, rel_points[near_plane] @ v))
    edges = np.roll(polygon_2d, -1, axis=0) - polygon_2d
    delta = point_2d[:, None, :] - polygon_2d[None, :, :]
    cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
    # 顶点已按法向保持逆时针顺序；凸多边形内点应位于每条有向边左侧。
    inside_2d = np.all(cross >= -tolerance, axis=1)
    result = np.zeros(len(points), dtype=bool)
    result[np.flatnonzero(near_plane)] = inside_2d
    return result


def voxelize_periodic_sheets(
    faces: Sequence[object],
    selected_ids: Sequence[int] | None,
    cube_size_nm: float,
    thickness_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    """Voxelize the physical union of periodic, finite-thickness sheet faces.

    The returned Boolean array is indexed in ``(x, y, z)`` order.  Its spacing
    is :func:`effective_voxel_size_nm`; the requested spacing is reduced just
    enough for an integer number of voxels to exactly fill the basic cell.

    Rather than double-counting individual prisms, every sheet image that can
    intersect the basic cell is tested and OR-ed into one occupancy grid.  This
    is equivalent to applying the minimum-image convention and correctly
    handles sheets crossing one or more periodic boundaries.

    对每个逻辑面枚举所有可能进入基本胞元的周期镜像，并将有限厚度棱柱的占据
    状态以逻辑或写入同一布尔数组。因此片层交叠处不会重复计体积，跨越一个或
    多个周期边界的片层也会在相对边界上生成严格对应的部分。
    """

    box_size = float(cube_size_nm)
    thickness = float(thickness_nm)
    if box_size <= 0.0 or thickness <= 0.0:
        raise ValueError("cube_size_nm and thickness_nm must be positive")
    spacing = effective_voxel_size_nm(box_size, float(voxel_size_nm))
    count = int(round(box_size / spacing))
    occupancy = np.zeros((count, count, count), dtype=bool)
    half_thickness = 0.5 * thickness
    tolerance = max(box_size * 1e-10, np.finfo(float).eps * box_size * 32.0)

    for face in _selected_faces(faces, selected_ids):
        polygon, normal = ordered_face_vertices(face)
        # 法向拉伸 t/2 后，在三个坐标方向上的包围盒增量为
        # (t/2)*abs(normal)，据此只检查可能相交的周期副本和局部体素块。
        extent = half_thickness * np.abs(normal)
        ranges = _translation_ranges(polygon, extent, box_size)
        for shift_index in product(*ranges):
            shift = box_size * np.asarray(shift_index, dtype=float)
            image = polygon + shift
            lo = image.min(axis=0) - extent
            hi = image.max(axis=0) + extent
            index_ranges = [
                _voxel_index_range(float(lo[axis]), float(hi[axis]), spacing, count)
                for axis in range(3)
            ]
            if any(first > last for first, last in index_ranges):
                continue
            axes = [
                (np.arange(first, last + 1, dtype=float) + 0.5) * spacing
                for first, last in index_ranges
            ]
            x, y, z = np.meshgrid(*axes, indexing="ij")
            points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            inside = _points_inside_extruded_polygon(
                points,
                image,
                normal,
                half_thickness,
                tolerance,
            ).reshape(x.shape)
            slices = tuple(slice(first, last + 1) for first, last in index_ranges)
            # 对所有片层及其周期镜像取并集，保证重叠体积仅计数一次。
            occupancy[slices] |= inside
    return occupancy


def mesh_from_occupancy(
    occupancy: np.ndarray,
    voxel_size_nm: float | None = None,
    *,
    cube_size_nm: float | None = None,
    origin_nm: Sequence[float] = (0.0, 0.0, 0.0),
    method: str = "lewiner",
    level: float = 0.5,
) -> TriangleMesh:
    """Extract a closed surface using marching cubes and one zero ghost layer.

    Padding by an empty layer makes pieces cut by the unit-cell boundary close
    exactly on that boundary.  If ``cube_size_nm`` is supplied, spacing is
    derived from the array shape and that size; otherwise ``voxel_size_nm`` is
    used directly.

    在占据数组外补一层空体素，marching cubes 会在基本胞元边界处生成封口面，
    而不是留下开放截面；随后把等值面坐标平移回真实胞元边界。这样导出的单个
    基本胞元是可直接导入求解器的封闭实体表面，同时其边界形貌仍可周期拼接。
    """

    if method not in {"lewiner", "lorensen"}:
        raise ValueError("method must be 'lewiner' or 'lorensen'")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    solid = np.asarray(occupancy, dtype=bool)
    if solid.ndim != 3 or min(solid.shape) < 1:
        raise ValueError("occupancy must be a non-empty 3D array")
    if cube_size_nm is not None:
        box = float(cube_size_nm)
        if box <= 0.0:
            raise ValueError("cube_size_nm must be positive")
        spacing = np.asarray([box / size for size in solid.shape], dtype=float)
    elif voxel_size_nm is not None:
        value = float(voxel_size_nm)
        if value <= 0.0:
            raise ValueError("voxel_size_nm must be positive")
        spacing = np.full(3, value, dtype=float)
    else:
        raise ValueError("provide voxel_size_nm or cube_size_nm")

    if not np.any(solid):
        return TriangleMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))

    # 空白幽灵层让所有被盒边界截断的固相在边界位置形成三角形封口。
    padded = np.pad(solid.astype(np.uint8), 1, mode="constant", constant_values=0)
    vertices, triangles, _, _ = marching_cubes(
        padded,
        level=float(level),
        spacing=tuple(float(value) for value in spacing),
        allow_degenerate=False,
        method=method,
    )
    origin = np.asarray(origin_nm, dtype=float)
    if origin.shape != (3,):
        raise ValueError("origin_nm must contain three coordinates")
    # 数组采样点代表体素中心。level=0.5 时，真实层和幽灵层之间的等值面恰好
    # 落在盒边界；最后的截断可抑制浮点误差，保证封口严格位于基本胞元表面。
    # 其他 level 仅供可选的全局体积校准使用。
    vertices += origin - 0.5 * spacing
    if cube_size_nm is not None:
        upper = origin + float(cube_size_nm)
        vertices = np.minimum(np.maximum(vertices, origin), upper)

    mesh = TriangleMesh(vertices, triangles)
    if mesh_signed_volume_nm3(mesh) < 0.0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
    return mesh


def _coerce_mesh(
    mesh_or_vertices: TriangleMesh | Mapping[str, object] | np.ndarray,
    faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh_or_vertices, TriangleMesh):
        return mesh_or_vertices.vertices_nm, mesh_or_vertices.faces
    if isinstance(mesh_or_vertices, Mapping):
        vertices = mesh_or_vertices.get("vertices_nm", mesh_or_vertices.get("vertices"))
        connectivity = mesh_or_vertices.get("faces", mesh_or_vertices.get("triangles"))
        if vertices is None or connectivity is None:
            raise ValueError("mesh mapping needs vertices_nm and faces/triangles")
        return np.asarray(vertices, dtype=float), np.asarray(connectivity, dtype=np.int64)
    if faces is None:
        raise ValueError("faces must be supplied with a vertex array")
    return np.asarray(mesh_or_vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def mesh_signed_volume_nm3(
    mesh_or_vertices: TriangleMesh | Mapping[str, object] | np.ndarray,
    faces: np.ndarray | None = None,
) -> float:
    """Return the oriented volume enclosed by a triangular surface."""

    vertices, triangles = _coerce_mesh(mesh_or_vertices, faces)
    if len(triangles) == 0:
        return 0.0
    tri = vertices[triangles]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def mesh_volume_nm3(
    mesh_or_vertices: TriangleMesh | Mapping[str, object] | np.ndarray,
    faces: np.ndarray | None = None,
) -> float:
    """Return the absolute enclosed volume in cubic nanometres."""

    return abs(mesh_signed_volume_nm3(mesh_or_vertices, faces))


def mesh_quality(
    mesh_or_vertices: TriangleMesh | Mapping[str, object] | np.ndarray,
    faces: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute volume and edge-based water-tightness diagnostics.

    通过统计每条无向边的出现次数以及两个三角形对该边的方向，检查边界边、
    非流形边和朝向不一致边；只有每条边恰好由两个反向三角形共享才判为水密。
    """

    vertices, triangles = _coerce_mesh(mesh_or_vertices, faces)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangle connectivity must have shape (M, 3)")
    if len(triangles) == 0:
        return {
            "n_vertices": int(len(vertices)),
            "n_faces": 0,
            "surface_area_nm2": 0.0,
            "signed_volume_nm3": 0.0,
            "volume_nm3": 0.0,
            "is_watertight": False,
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "inconsistent_oriented_edge_count": 0,
            "bounds_nm": None,
        }

    # 同时保留有向边和排序后的无向边：前者用于检查朝向，后者用于计数邻面。
    directed = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    undirected = np.sort(directed, axis=1)
    unique_edges, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    forward = directed[:, 0] == undirected[:, 0]
    forward_count = np.bincount(inverse, weights=forward.astype(np.int64), minlength=len(unique_edges))
    two_sided = counts == 2
    inconsistent = two_sided & (forward_count != 1)

    tri = vertices[triangles]
    double_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    signed_volume = mesh_signed_volume_nm3(vertices, triangles)
    return {
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(triangles)),
        "surface_area_nm2": float(0.5 * double_area.sum()),
        "signed_volume_nm3": float(signed_volume),
        "volume_nm3": float(abs(signed_volume)),
        "is_watertight": bool(np.all(counts == 2) and not np.any(inconsistent)),
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "inconsistent_oriented_edge_count": int(np.count_nonzero(inconsistent)),
        "bounds_nm": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def write_binary_stl(
    path: str | Path,
    mesh_or_vertices: TriangleMesh | Mapping[str, object] | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    header: str = "Periodic Voronoi aerogel; coordinates in nm",
) -> Path:
    """Write a binary STL without a third-party mesh dependency.

    STL 不记录长度单位，本项目约定文件中的数值坐标为 nm；Lumerical 导入脚本
    会使用 1e-9 比例换算为 SI 米制。分块写出可避免大型网格产生额外峰值内存。
    """

    vertices, triangles = _coerce_mesh(mesh_or_vertices, faces)
    if len(triangles) > np.iinfo(np.uint32).max:
        raise ValueError("binary STL supports at most 2^32-1 triangles")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("face indices are outside the vertex array")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or infinite values")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = header.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attribute", "<u2"),
        ]
    )
    with target.open("wb") as stream:
        stream.write(header_bytes)
        stream.write(struct.pack("<I", len(triangles)))
        # 大模型按块计算法向并写盘，避免一次复制全部三角形顶点。
        chunk_size = 100_000
        for start in range(0, len(triangles), chunk_size):
            connectivity = triangles[start : start + chunk_size]
            tri = vertices[connectivity]
            raw_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            lengths = np.linalg.norm(raw_normals, axis=1)
            normals = np.zeros_like(raw_normals)
            valid = lengths > np.finfo(float).eps
            normals[valid] = raw_normals[valid] / lengths[valid, None]
            records = np.zeros(len(tri), dtype=record_dtype)
            records["normal"] = normals.astype(np.float32)
            records["v0"] = tri[:, 0].astype(np.float32)
            records["v1"] = tri[:, 1].astype(np.float32)
            records["v2"] = tri[:, 2].astype(np.float32)
            stream.write(records.tobytes())
    return target


__all__ = [
    "TriangleMesh",
    "effective_voxel_size_nm",
    "face_normal",
    "face_vertices_nm",
    "mesh_from_occupancy",
    "mesh_quality",
    "mesh_signed_volume_nm3",
    "mesh_volume_nm3",
    "ordered_face_vertices",
    "voxelize_periodic_sheets",
    "write_binary_stl",
]
