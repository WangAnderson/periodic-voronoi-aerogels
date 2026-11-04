"""Periodic three-dimensional Voronoi geometry.

The module keeps topology in an unwrapped periodic covering space.  Geometry is
only translated and clipped to the primary box when :func:`periodic_face_patches`
is called.  This distinction is important: clipping faces before building the
topology loses connections that cross a periodic boundary.

本模块在“展开的周期覆盖空间”中保存拓扑：跨越边界的面仍是一个完整逻辑面，
只在导出主单元时才平移并裁剪。若先按立方体边界裁剪，会破坏周期面之间的真实连接关系。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor, pi
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import QhullError, Voronoi


FloatArray: TypeAlias = NDArray[np.float64]
Int3: TypeAlias = tuple[int, int, int]
FaceKey: TypeAlias = tuple[int, int, int, int, int]
EdgeKey: TypeAlias = tuple[int, int, int, int, int]

_TILE_SHIFTS = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=np.int8)
_CENTRAL_TILE = int(np.flatnonzero(np.all(_TILE_SHIFTS == 0, axis=1))[0])


@dataclass(slots=True)
class PeriodicFace:
    """One canonical Voronoi face in the periodic covering space.

    ``shift`` is the integer box translation from ``cell_a`` to the image of
    ``cell_b`` sharing this face.  The normal points in that same direction.
    ``vertices_nm`` are ordered counter-clockwise when viewed along ``normal``
    and are intentionally not wrapped into the primary box.

    每个物理公共面只保留一个规范表示。``shift`` 表示从 ``cell_a`` 到与其
    共面的 ``cell_b`` 周期像所跨越的整数盒子位移；顶点故意不折回主单元，
    以便保留连续的平面多边形。
    """

    id: int
    cell_a: int
    cell_b: int
    shift: Int3
    vertices_nm: FloatArray
    normal: FloatArray
    area_nm2: float
    centroid_nm: FloatArray
    equivalent_diameter_nm: float
    min_edge_nm: float

    @property
    def key(self) -> FaceKey:
        return (self.cell_a, self.cell_b, *self.shift)


@dataclass(frozen=True, slots=True)
class FaceAdjacency:
    """Connection between two faces through a periodic Voronoi edge.

    Translating ``face_b`` by ``shift * box_size`` aligns its shared edge with
    the occurrence stored by ``face_a``.

    这是固相片层图的邻接边：两个逻辑面共用一条周期 Voronoi 边时相连。
    ``shift`` 记录对齐两份共边所需的整数周期平移。
    """

    face_a: int
    face_b: int
    shift: Int3
    edge_key: EdgeKey


@dataclass(frozen=True, slots=True)
class PoreEdge:
    """Periodic dual-graph edge through an unblocked Voronoi face.

    未被片层占据的 Voronoi 面对应孔相对偶图中的一条通道。
    """

    face_id: int
    cell_a: int
    cell_b: int
    shift: Int3


@dataclass(slots=True)
class PeriodicVoronoi:
    """Canonical faces and graph topology for a periodic tessellation.

    集中保存主单元种子、规范逻辑面、固相面邻接关系与孔相对偶边。
    """

    box_size_nm: float
    seeds_nm: FloatArray
    faces: list[PeriodicFace]
    face_adjacency: list[FaceAdjacency]
    pore_edges: list[PoreEdge]


def generate_periodic_seeds(
    num_seeds: int,
    box_size_nm: float,
    *,
    min_distance_factor: float = 0.32,
    random_seed: int | np.random.Generator | None = None,
    max_attempts_per_seed: int = 200,
) -> FloatArray:
    """Generate random sequential-inhibition points on a three-torus.

    The enforced minimum distance is
    ``min_distance_factor * box_size_nm / cbrt(num_seeds)``.  The result is
    "Poisson-ish": it suppresses near-coincident points without performing an
    expensive centroidal relaxation, and it always contains exactly
    ``num_seeds`` points or raises ``RuntimeError``.

    候选点之间采用最小镜像距离，因此立方体相对两侧也遵守同一最小间距。
    空间分桶只用于加速邻点查询，不会改变随机抑制采样的几何定义。
    """

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if box_size_nm <= 0.0:
        raise ValueError("box_size_nm must be positive")
    if min_distance_factor < 0.0:
        raise ValueError("min_distance_factor must be non-negative")
    if max_attempts_per_seed <= 0:
        raise ValueError("max_attempts_per_seed must be positive")

    rng = (
        random_seed
        if isinstance(random_seed, np.random.Generator)
        else np.random.default_rng(random_seed)
    )
    if min_distance_factor == 0.0:
        return rng.random((num_seeds, 3)) * box_size_nm

    min_distance = min_distance_factor * box_size_nm / np.cbrt(num_seeds)
    bins_per_axis = max(1, int(floor(box_size_nm / min_distance)))
    bin_size = box_size_nm / bins_per_axis
    neighbour_offsets = tuple(product((-1, 0, 1), repeat=3))

    points = np.empty((num_seeds, 3), dtype=np.float64)
    bins: dict[Int3, list[int]] = {}
    accepted = 0
    attempts = 0
    max_attempts = max_attempts_per_seed * num_seeds
    min_distance_sq = min_distance * min_distance

    while accepted < num_seeds and attempts < max_attempts:
        attempts += 1
        candidate = rng.random(3) * box_size_nm
        base_bin_array = np.floor(candidate / bin_size).astype(np.int64)
        base_bin = tuple(int(value) for value in base_bin_array)
        nearby_indices: list[int] = []
        visited_bins: set[Int3] = set()
        for offset in neighbour_offsets:
            neighbour_bin = tuple(
                (base_bin[axis] + offset[axis]) % bins_per_axis
                for axis in range(3)
            )
            if neighbour_bin in visited_bins:
                continue
            visited_bins.add(neighbour_bin)
            nearby_indices.extend(bins.get(neighbour_bin, ()))

        if nearby_indices:
            delta = np.abs(points[nearby_indices] - candidate)
            delta = np.minimum(delta, box_size_nm - delta)
            if np.any(np.einsum("ij,ij->i", delta, delta) < min_distance_sq):
                continue

        points[accepted] = candidate
        bins.setdefault(base_bin, []).append(accepted)
        accepted += 1

    if accepted != num_seeds:
        raise RuntimeError(
            "Could not place all periodic seeds with the requested minimum "
            "distance; lower min_distance_factor or raise max_attempts_per_seed"
        )
    return points


def build_periodic_voronoi(
    seeds_nm: ArrayLike,
    box_size_nm: float,
    *,
    tolerance_nm: float | None = None,
    qhull_options: str | None = None,
) -> PeriodicVoronoi:
    """Build a 3-D periodic Voronoi tessellation from 27 tiled seed copies.

    Parameters
    ----------
    seeds_nm:
        Base seed coordinates.  Coordinates are reduced modulo ``box_size_nm``.
    box_size_nm:
        Side length of the cubic periodic box.
    tolerance_nm:
        Coordinate tolerance used only for periodic topological de-duplication.
        It defaults to ``1e-9 * box_size_nm``.
    qhull_options:
        Optional options passed directly to :class:`scipy.spatial.Voronoi`.

    Returns
    -------
    PeriodicVoronoi
        Canonical, unwrapped faces plus face and cell graph edges.

    实现要点：将主单元种子复制到 3×3×3 邻域中计算 Voronoi 图，仅收集
    与中心副本相关的面，再用 ``(cell_a, cell_b, shift)`` 将正反方向及周期
    副本合并为唯一逻辑面。
    """

    if box_size_nm <= 0.0:
        raise ValueError("box_size_nm must be positive")
    seeds = np.asarray(seeds_nm, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or len(seeds) == 0:
        raise ValueError("seeds_nm must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(seeds)):
        raise ValueError("seeds_nm must contain only finite values")
    seeds = np.mod(seeds, box_size_nm)
    tolerance = _resolve_tolerance(box_size_nm, tolerance_nm)
    _check_unique_periodic_seeds(seeds, box_size_nm, tolerance)

    num_seeds = len(seeds)
    # 27 个副本使中心单元的每个邻居都能通过普通 Voronoi 计算找到。
    tiled_points = (
        seeds[None, :, :]
        + _TILE_SHIFTS[:, None, :].astype(np.float64) * box_size_nm
    ).reshape(-1, 3)
    try:
        diagram = Voronoi(tiled_points, qhull_options=qhull_options)
    except QhullError as exc:
        raise RuntimeError(
            "Qhull could not construct the tiled 3-D Voronoi diagram; check "
            "for degenerate seeds or retry with a deterministic small jitter"
        ) from exc

    central = np.zeros(len(tiled_points), dtype=bool)
    central[
        _CENTRAL_TILE * num_seeds : (_CENTRAL_TILE + 1) * num_seeds
    ] = True

    raw_faces: dict[FaceKey, FloatArray] = {}
    for ridge_index, (point_a, point_b) in enumerate(diagram.ridge_points):
        # 只保留至少有一端属于中心副本的脊，避免重复处理外围副本。
        if not (central[point_a] or central[point_b]):
            continue
        ridge_vertices = diagram.ridge_vertices[ridge_index]
        if -1 in ridge_vertices:
            raise RuntimeError(
                "A ridge incident to a central periodic cell is unbounded; "
                "the 3x3x3 tiling invariant was violated"
            )

        if central[point_a]:
            centre_index, other_index = int(point_a), int(point_b)
        else:
            centre_index, other_index = int(point_b), int(point_a)
        cell_i = centre_index % num_seeds
        cell_j = other_index % num_seeds
        relative_shift = (
            _TILE_SHIFTS[other_index // num_seeds]
            - _TILE_SHIFTS[centre_index // num_seeds]
        ).astype(np.int64)
        vertices = diagram.vertices[np.asarray(ridge_vertices, dtype=np.int64)]

        key, canonical_vertices = _canonical_face(
            cell_i,
            cell_j,
            relative_shift,
            vertices,
            box_size_nm,
        )
        previous = raw_faces.get(key)
        # 同一公共面可能从相邻胞元的相反方向被发现，规范键保证只保留一份。
        if previous is None:
            raw_faces[key] = canonical_vertices.copy()
        elif not _same_point_set(previous, canonical_vertices, tolerance):
            raise RuntimeError(f"Inconsistent duplicate periodic face for key {key}")

    # Qhull's tiled diagram dominates memory.  Everything needed below has
    # already been copied into canonical face polygons.
    del diagram, tiled_points

    faces: list[PeriodicFace] = []
    for face_id, key in enumerate(sorted(raw_faces)):
        cell_a, cell_b, shift_x, shift_y, shift_z = key
        shift = (shift_x, shift_y, shift_z)
        neighbour_position = seeds[cell_b] + np.asarray(shift) * box_size_nm
        normal = neighbour_position - seeds[cell_a]
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= tolerance:
            raise RuntimeError(f"Degenerate face normal for key {key}")
        normal /= normal_length
        vertices = _order_face_vertices(raw_faces[key], normal, tolerance)
        area, centroid = _polygon_area_centroid(vertices, normal)
        if area <= tolerance * tolerance:
            raise RuntimeError(f"Degenerate periodic face for key {key}")
        edge_lengths = np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)
        faces.append(
            PeriodicFace(
                id=face_id,
                cell_a=cell_a,
                cell_b=cell_b,
                shift=shift,
                vertices_nm=vertices,
                normal=normal,
                area_nm2=area,
                centroid_nm=centroid,
                equivalent_diameter_nm=2.0 * np.sqrt(area / pi),
                min_edge_nm=float(np.min(edge_lengths)),
            )
        )

    face_adjacency = _build_face_adjacency(faces, box_size_nm, tolerance)
    pore_edges = [
        PoreEdge(
            face_id=face.id,
            cell_a=face.cell_a,
            cell_b=face.cell_b,
            shift=face.shift,
        )
        for face in faces
    ]
    return PeriodicVoronoi(
        box_size_nm=float(box_size_nm),
        seeds_nm=seeds,
        faces=faces,
        face_adjacency=face_adjacency,
        pore_edges=pore_edges,
    )


def periodic_face_patches(
    face: PeriodicFace,
    box_size_nm: float,
    *,
    tolerance_nm: float | None = None,
) -> list[FloatArray]:
    """Translate and clip one unwrapped face into the primary cubic box.

    将一个完整周期逻辑面转换为主单元内可导出的一个或多个裁剪面片。
    """

    return translate_and_clip_periodic_polygon(
        face.vertices_nm,
        box_size_nm,
        tolerance_nm=tolerance_nm,
    )


def translate_and_clip_periodic_polygon(
    vertices_nm: ArrayLike,
    box_size_nm: float,
    *,
    tolerance_nm: float | None = None,
) -> list[FloatArray]:
    """Return every translated piece of a planar polygon inside ``[0, L]^3``.

    The input polygon may use any periodic lift and may cross more than one box
    boundary.  Each returned array is an ordered convex polygon.  This function
    clips a zero-thickness face; a thick sheet must instead be thickened in the
    covering space before its translated copies are clipped.

    返回值仅是几何面片；“一个逻辑面要整体保留或删除”的选择仍在裁剪之前完成。
    """

    if box_size_nm <= 0.0:
        raise ValueError("box_size_nm must be positive")
    vertices = np.asarray(vertices_nm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("vertices_nm must have shape (M, 3) with M >= 3")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices_nm must contain only finite values")
    tolerance = _resolve_tolerance(box_size_nm, tolerance_nm)

    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    shift_ranges: list[range] = []
    for axis in range(3):
        first = int(ceil((-maximum[axis] - tolerance) / box_size_nm))
        last = int(floor((box_size_nm - minimum[axis] + tolerance) / box_size_nm))
        shift_ranges.append(range(first, last + 1))

    patches: list[FloatArray] = []
    for tile_shift in product(*shift_ranges):
        translated = vertices + np.asarray(tile_shift, dtype=np.float64) * box_size_nm
        clipped = _clip_polygon_to_box(translated, box_size_nm, tolerance)
        if len(clipped) < 3:
            continue
        area_vector = 0.5 * np.sum(
            np.cross(clipped, np.roll(clipped, -1, axis=0)), axis=0
        )
        if float(np.linalg.norm(area_vector)) <= tolerance * tolerance:
            continue
        patches.append(clipped)
    return patches


def _resolve_tolerance(box_size_nm: float, tolerance_nm: float | None) -> float:
    if tolerance_nm is None:
        return max(1.0e-9 * box_size_nm, np.finfo(np.float64).eps * box_size_nm * 100)
    if tolerance_nm <= 0.0:
        raise ValueError("tolerance_nm must be positive")
    return float(tolerance_nm)


def _check_unique_periodic_seeds(
    seeds: FloatArray,
    box_size_nm: float,
    tolerance: float,
) -> None:
    # Sorting is unnecessary here: generated seed sets are modest (<= a few
    # thousand), and a periodic spatial bin avoids an O(N^2) distance matrix.
    # 比较距离时使用最小镜像，否则会漏掉分别靠近 0 与 L 边界的重复种子。
    bins_per_axis = max(1, int(floor(box_size_nm / tolerance)))
    bins: dict[Int3, list[int]] = {}
    for index, point in enumerate(seeds):
        scaled = np.floor(point / box_size_nm * bins_per_axis).astype(np.int64)
        base_bin = tuple(int(value) for value in scaled)
        visited_bins: set[Int3] = set()
        for offset in product((-1, 0, 1), repeat=3):
            neighbour_bin = tuple(
                (base_bin[axis] + offset[axis]) % bins_per_axis
                for axis in range(3)
            )
            if neighbour_bin in visited_bins:
                continue
            visited_bins.add(neighbour_bin)
            for other in bins.get(neighbour_bin, ()):
                delta = np.abs(point - seeds[other])
                delta = np.minimum(delta, box_size_nm - delta)
                if float(np.linalg.norm(delta)) <= tolerance:
                    raise ValueError("seeds_nm contains duplicate periodic points")
        bins.setdefault(base_bin, []).append(index)


def _canonical_face(
    cell_i: int,
    cell_j: int,
    relative_shift: NDArray[np.int64],
    vertices: FloatArray,
    box_size_nm: float,
) -> tuple[FaceKey, FloatArray]:
    """将公共面及其反向表示合并为同一规范键。

    ``(i, j, s)`` 与 ``(j, i, -s)`` 表示同一物理面。当胞元顺序反转时，
    同时平移顶点，使其仍处在新 ``cell_a`` 所在的展开坐标分支中。
    """

    shift = tuple(int(value) for value in relative_shift)
    if cell_i < cell_j:
        return (cell_i, cell_j, *shift), vertices
    if cell_i > cell_j:
        reverse_shift = tuple(-value for value in shift)
        return (
            (cell_j, cell_i, *reverse_shift),
            vertices - relative_shift * box_size_nm,
        )

    reverse_shift = tuple(-value for value in shift)
    if reverse_shift < shift:
        return (
            (cell_i, cell_i, *reverse_shift),
            vertices - relative_shift * box_size_nm,
        )
    return (cell_i, cell_i, *shift), vertices


def _same_point_set(points_a: FloatArray, points_b: FloatArray, tolerance: float) -> bool:
    if len(points_a) != len(points_b):
        return False
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    return bool(
        np.max(np.min(distances, axis=0)) <= tolerance
        and np.max(np.min(distances, axis=1)) <= tolerance
    )


def _order_face_vertices(
    vertices: FloatArray,
    normal: FloatArray,
    tolerance: float,
) -> FloatArray:
    unique: list[FloatArray] = []
    for vertex in vertices:
        if not any(np.linalg.norm(vertex - previous) <= tolerance for previous in unique):
            unique.append(vertex)
    if len(unique) < 3:
        raise RuntimeError("A Voronoi face has fewer than three unique vertices")

    points = np.asarray(unique, dtype=np.float64)
    centre = np.mean(points, axis=0)
    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    basis_u = np.cross(normal, reference_axis)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    offsets = points - centre
    angles = np.arctan2(offsets @ basis_v, offsets @ basis_u)
    ordered = points[np.argsort(angles)]

    area_vector = 0.5 * np.sum(
        np.cross(ordered, np.roll(ordered, -1, axis=0)), axis=0
    )
    if float(np.dot(area_vector, normal)) < 0.0:
        ordered = ordered[::-1].copy()
    return ordered


def _polygon_area_centroid(
    vertices: FloatArray,
    normal: FloatArray,
) -> tuple[float, FloatArray]:
    origin = vertices[0]
    weighted_centroid = np.zeros(3, dtype=np.float64)
    total_area = 0.0
    for index in range(1, len(vertices) - 1):
        point_b = vertices[index]
        point_c = vertices[index + 1]
        twice_signed_area = float(
            np.dot(np.cross(point_b - origin, point_c - origin), normal)
        )
        triangle_area = 0.5 * twice_signed_area
        total_area += triangle_area
        weighted_centroid += triangle_area * (origin + point_b + point_c) / 3.0
    if total_area <= 0.0:
        return 0.0, np.mean(vertices, axis=0)
    return total_area, weighted_centroid / total_area


class _PeriodicVertexRegistry:
    """将展开空间中的顶点登记为“主单元顶点 ID + 整数周期位移”。

    这使相对边界上的几何等价点共用同一 ID，同时仍可由 ``lift`` 重建原坐标。
    """

    def __init__(self, box_size_nm: float, tolerance_nm: float) -> None:
        self.box_size_nm = box_size_nm
        self.tolerance_nm = tolerance_nm
        self.positions: list[FloatArray] = []
        self.buckets: dict[Int3, list[int]] = {}

    def register(self, point: FloatArray) -> tuple[int, NDArray[np.int64]]:
        """返回周期唯一顶点 ID 以及该点所在的整数周期层。"""

        wrapped = np.mod(point, self.box_size_nm)
        wrapped[np.abs(wrapped) <= self.tolerance_nm] = 0.0
        wrapped[
            np.abs(wrapped - self.box_size_nm) <= self.tolerance_nm
        ] = 0.0
        bucket = tuple(
            int(value)
            for value in np.rint(wrapped / self.tolerance_nm).astype(np.int64)
        )
        vertex_id: int | None = None
        for offset in product((-1, 0, 1), repeat=3):
            neighbour_bucket = tuple(bucket[axis] + offset[axis] for axis in range(3))
            for candidate_id in self.buckets.get(neighbour_bucket, ()):
                delta = np.abs(wrapped - self.positions[candidate_id])
                delta = np.minimum(delta, self.box_size_nm - delta)
                if float(np.linalg.norm(delta)) <= self.tolerance_nm:
                    vertex_id = candidate_id
                    wrapped = self.positions[candidate_id]
                    break
            if vertex_id is not None:
                break
        if vertex_id is None:
            vertex_id = len(self.positions)
            self.positions.append(wrapped.copy())
            self.buckets.setdefault(bucket, []).append(vertex_id)

        lift = np.rint((point - wrapped) / self.box_size_nm).astype(np.int64)
        reconstruction_error = np.linalg.norm(
            point - (wrapped + lift * self.box_size_nm)
        )
        if reconstruction_error > 2.0 * self.tolerance_nm:
            raise RuntimeError("Could not assign a consistent periodic vertex lift")
        return vertex_id, lift


def _build_face_adjacency(
    faces: list[PeriodicFace],
    box_size_nm: float,
    tolerance: float,
) -> list[FaceAdjacency]:
    """通过共用周期 Voronoi 边建立逻辑面邻接图。

    边键由两个周期顶点 ID 及其相对 ``lift`` 构成，因此被立方体边界分开的
    共边仍能正确合并。
    """

    registry = _PeriodicVertexRegistry(box_size_nm, tolerance)
    edge_incidents: dict[EdgeKey, list[tuple[int, FloatArray]]] = {}

    for face in faces:
        references = [registry.register(vertex) for vertex in face.vertices_nm]
        for index, vertex in enumerate(face.vertices_nm):
            next_index = (index + 1) % len(face.vertices_nm)
            vertex_a, lift_a = references[index]
            vertex_b, lift_b = references[next_index]
            relative_lift = lift_b - lift_a
            # 按顶点 ID 排序，并在反向时同步取反位移，使无向边只有一个键。
            if vertex_a < vertex_b:
                edge_key = (vertex_a, vertex_b, *map(int, relative_lift))
            elif vertex_b < vertex_a:
                edge_key = (vertex_b, vertex_a, *map(int, -relative_lift))
            else:
                winding = tuple(int(value) for value in relative_lift)
                reverse_winding = tuple(-value for value in winding)
                canonical_winding = min(winding, reverse_winding)
                edge_key = (vertex_a, vertex_a, *canonical_winding)
            midpoint = 0.5 * (vertex + face.vertices_nm[next_index])
            edge_incidents.setdefault(edge_key, []).append((face.id, midpoint))

    adjacency: list[FaceAdjacency] = []
    for edge_key, incidents in sorted(edge_incidents.items()):
        by_face: dict[int, FloatArray] = {}
        for face_id, midpoint in incidents:
            by_face.setdefault(face_id, midpoint)
        ordered_incidents = sorted(by_face.items())
        if len(ordered_incidents) < 2:
            raise RuntimeError(f"Periodic Voronoi edge {edge_key} has one incident face")
        for index_a in range(len(ordered_incidents)):
            face_a, midpoint_a = ordered_incidents[index_a]
            for index_b in range(index_a + 1, len(ordered_incidents)):
                face_b, midpoint_b = ordered_incidents[index_b]
                shift_array = np.rint(
                    (midpoint_a - midpoint_b) / box_size_nm
                ).astype(np.int64)
                # 两个面保存的共边可能位于不同周期层；中点差给出对齐位移。
                aligned_error = np.linalg.norm(
                    midpoint_b + shift_array * box_size_nm - midpoint_a
                )
                if aligned_error > 2.0 * tolerance:
                    raise RuntimeError(
                        f"Could not align shared periodic edge {edge_key}"
                    )
                adjacency.append(
                    FaceAdjacency(
                        face_a=face_a,
                        face_b=face_b,
                        shift=tuple(int(value) for value in shift_array),
                        edge_key=edge_key,
                    )
                )
    return adjacency


def _clip_polygon_to_box(
    vertices: FloatArray,
    box_size_nm: float,
    tolerance: float,
) -> FloatArray:
    polygon = vertices.copy()
    for axis in range(3):
        polygon = _clip_polygon_halfspace(
            polygon,
            axis,
            bound=0.0,
            keep_above=True,
            tolerance=tolerance,
        )
        polygon = _clip_polygon_halfspace(
            polygon,
            axis,
            bound=box_size_nm,
            keep_above=False,
            tolerance=tolerance,
        )
        if len(polygon) == 0:
            return polygon

    polygon[np.abs(polygon) <= tolerance] = 0.0
    polygon[np.abs(polygon - box_size_nm) <= tolerance] = box_size_nm
    return _remove_consecutive_duplicates(polygon, tolerance)


def _clip_polygon_halfspace(
    vertices: FloatArray,
    axis: int,
    *,
    bound: float,
    keep_above: bool,
    tolerance: float,
) -> FloatArray:
    if len(vertices) == 0:
        return vertices

    def inside(point: FloatArray) -> bool:
        if keep_above:
            return bool(point[axis] >= bound - tolerance)
        return bool(point[axis] <= bound + tolerance)

    output: list[FloatArray] = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > np.finfo(np.float64).eps:
                fraction = (bound - previous[axis]) / denominator
                intersection = previous + fraction * (current - previous)
                intersection[axis] = bound
                output.append(intersection)
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _remove_consecutive_duplicates(
    vertices: FloatArray,
    tolerance: float,
) -> FloatArray:
    if len(vertices) < 2:
        return vertices
    unique = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - unique[-1]) > tolerance:
            unique.append(vertex)
    if len(unique) > 1 and np.linalg.norm(unique[0] - unique[-1]) <= tolerance:
        unique.pop()
    return np.asarray(unique, dtype=np.float64)


__all__ = [
    "FaceAdjacency",
    "PeriodicFace",
    "PeriodicVoronoi",
    "PoreEdge",
    "build_periodic_voronoi",
    "generate_periodic_seeds",
    "periodic_face_patches",
    "translate_and_clip_periodic_polygon",
]
