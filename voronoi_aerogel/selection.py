"""Face selection for a periodic, bicontinuous Voronoi sheet network.

The module consumes geometry by duck typing.  A face only needs the fields
documented by :class:`~voronoi_aerogel.geometry.PeriodicFace`; no geometry
module is imported here.  Logical faces are addressed by their index in the
input sequence, so all fragments of a boundary-crossing face are selected as
one object.

本模块负责从周期 Voronoi 公共面中选出有限厚度的片层。核心约束是：
固相和孔相都必须保持连通，并在 x/y/z 三个方向上具有独立的周期绕行路径。
跨边界的多个几何碎片共享同一个“逻辑面”编号，因此总是作为一个整体
保留或删除，从而保证基本单元能无缝周期平铺。
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from math import ceil, floor, pi
from typing import Callable, Hashable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .topology import (
    AxisBackbone,
    PeriodicGraph,
    WindingReport,
    analyze_winding,
    build_axis_backbone,
    connected_component,
    graph_from_adjacency,
)


class FaceLike(Protocol):
    normal: object
    shift: object


def _area(face: FaceLike) -> float:
    value = getattr(face, "area", None)
    if value is None:
        value = getattr(face, "area_nm2")
    return float(value)


def _diameter(face: FaceLike) -> float:
    value = getattr(face, "equivalent_diameter", None)
    if value is None:
        value = getattr(face, "equivalent_diameter_nm", None)
    if value is not None:
        return float(value)
    return 2.0 * np.sqrt(max(_area(face), 0.0) / pi)


def _vertices(face: FaceLike) -> np.ndarray:
    value = getattr(face, "vertices", None)
    if value is None:
        value = getattr(face, "vertices_nm")
    return np.asarray(value, dtype=np.float64)


def _cells(face: FaceLike) -> tuple[Hashable, Hashable]:
    value = getattr(face, "cells", None)
    if value is not None:
        cells = tuple(value)
    else:
        cells = (getattr(face, "cell_a"), getattr(face, "cell_b"))
    if len(cells) != 2:
        raise ValueError("a periodic face must reference exactly two cells")
    return cells[0], cells[1]


def build_pore_graph(faces: Sequence[FaceLike]) -> PeriodicGraph:
    """Build the periodic cell graph; edge payloads are logical face IDs.

    每个 Voronoi 胞元是孔相图的节点，两胞元间的公共面是边。如果某面未被
    选为固相，对应边就是开放的孔道；边上的整数 shift 记录它穿越了哪个
    周期边界。
    """

    graph = PeriodicGraph()
    for face_id, face in enumerate(faces):
        cells = _cells(face)
        graph.add_edge(
            cells[0],
            cells[1],
            getattr(face, "shift"),
            payload=face_id,
        )
    return graph


def coerce_face_graph(
    adjacency: PeriodicGraph
    | Mapping[Hashable, Iterable[object]]
    | Sequence[Iterable[object]],
    *,
    face_count: int | None = None,
) -> PeriodicGraph:
    """Accept a ready graph or a common adjacency-list representation."""

    if isinstance(adjacency, PeriodicGraph):
        graph = adjacency
    elif isinstance(adjacency, Sequence) and adjacency and hasattr(
        adjacency[0], "face_a"
    ):
        graph = PeriodicGraph(range(face_count or 0))
        for item in adjacency:
            graph.add_edge(
                getattr(item, "face_a"),
                getattr(item, "face_b"),
                getattr(item, "shift"),
                payload=getattr(item, "edge_key", None),
            )
    else:
        graph = graph_from_adjacency(adjacency)
    if face_count is not None:
        for face_id in range(face_count):
            graph.add_node(face_id)
    return graph


@dataclass(frozen=True, slots=True)
class DualBackbone:
    """固/孔双连通骨架及后续可生长的逻辑面集合。"""

    pore: AxisBackbone
    solid: AxisBackbone
    protected_face_ids: frozenset[int]
    solid_face_ids: frozenset[int]
    growable_face_ids: frozenset[int]
    nominal_min_solid_fraction: float
    nominal_max_solid_fraction_upper_bound: float


def _sample_roots(
    values: Iterable[Hashable],
    limit: int,
    rng: np.random.Generator,
) -> list[Hashable]:
    roots = list(values)
    if len(roots) <= limit:
        return roots
    indices = rng.choice(len(roots), size=limit, replace=False)
    return [roots[int(index)] for index in indices]


def _extend_pore_backbone_with_spanning_tree(
    graph: PeriodicGraph,
    backbone: AxisBackbone,
    *,
    item_cost: Callable[[Hashable], float],
) -> AxisBackbone:
    """Extend winding loops to a quotient-graph spanning tree.

    Unit-winding paths alone guarantee a percolating pore component but do not
    prevent other Voronoi cells from being sealed off later.  Protecting a
    minimum-cost spanning tree makes every pore cell part of that same open
    component while the existing winding paths retain rank three.

    仅保护三条周期绕行路径，只能证明“有一个”贯通孔道，仍可能把其他胞元
    封成孤立闭孔。因此再用 Kruskal 思路补齐一棵最小代价生成树：树上公共面
    永远不转为固相，使所有胞元都属于同一开放孔相；原有三向绕行路径
    则继续保证孔相的周期贯通性。
    """

    parent = {node: node for node in graph.nodes}
    rank = {node: 0 for node in graph.nodes}

    def find(node: Hashable) -> Hashable:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: Hashable, right: Hashable) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        return True

    # 先将三向周期路径加入并查集，补树时不会破坏已建立的绕行环。
    chosen_edges = set(backbone.edge_ids)
    chosen_items = set(backbone.items)
    for edge_id in chosen_edges:
        edge = graph.edge(edge_id)
        union(edge.source, edge.target)

    def edge_item(edge: object) -> Hashable:
        payload = getattr(edge, "payload")
        return payload if payload is not None else getattr(edge, "edge_id")

    # 按保护代价由低到高选边，得到连接全部胞元的最小代价树。
    ordered_edges = sorted(
        graph.iter_edges(),
        key=lambda edge: (float(item_cost(edge_item(edge))), edge.edge_id),
    )
    for edge in ordered_edges:
        if union(edge.source, edge.target):
            chosen_edges.add(edge.edge_id)
            chosen_items.add(edge_item(edge))

    roots = {find(node) for node in graph.nodes}
    if len(roots) > 1:
        raise RuntimeError("periodic pore graph is disconnected")
    winding = analyze_winding(graph, active_edge_ids=chosen_edges)
    return AxisBackbone(
        root=backbone.root,
        mode=backbone.mode,
        nodes=frozenset(graph.nodes),
        edge_ids=frozenset(chosen_edges),
        items=frozenset(chosen_items),
        paths=backbone.paths,
        cost=sum(float(item_cost(item)) for item in chosen_items),
        winding=winding,
    )


def construct_dual_backbone(
    faces: Sequence[FaceLike],
    face_graph: PeriodicGraph,
    *,
    cube_size_nm: float,
    thickness_nm: float,
    target_porosity: float,
    diameter_range_nm: tuple[float, float] | None = None,
    hard_diameter_range: bool = False,
    allowed_face_ids: Iterable[int] | None = None,
    offset_bound: int = 1,
    random_seed: int = 0,
    max_pore_roots: int = 8,
    max_solid_roots: int = 12,
) -> DualBackbone:
    """Construct disjoint unit-winding pore and solid backbones.

    Several pore roots are tried because the cheapest open-cell loops can
    conflict with all three solid loops.  The chosen pair first favours a
    nominal interval covering the requested solid volume, then a small solid
    skeleton and a small amount of protected face area.

    算法先建立必须保持开放的孔相骨架，再从剩余面中建立固相骨架。两者
    互不相交，且都含有三个独立的单位绕行方向。尝试多个起点是为了避免
    局部最便宜的孔道恰好占用了固相三向贯通所必需的面。
    """

    if not 0.0 < target_porosity < 1.0:
        raise ValueError("target_porosity must lie strictly between zero and one")
    if cube_size_nm <= 0 or thickness_nm <= 0:
        raise ValueError("cube_size_nm and thickness_nm must be positive")

    face_count = len(faces)
    expected_nodes = set(range(face_count))
    missing = expected_nodes.difference(face_graph.nodes)
    if missing:
        raise ValueError(
            "face_graph must contain every logical face node; "
            f"missing {len(missing)} nodes"
        )

    pore_graph = build_pore_graph(faces)
    rng = np.random.default_rng(random_seed)
    # area * thickness 是单面的名义体积；这里尚未扣除片层交叠区域。
    face_volume = np.asarray(
        [max(_area(face), 0.0) * thickness_nm for face in faces],
        dtype=np.float64,
    )
    box_volume = float(cube_size_nm) ** 3
    target_volume = (1.0 - target_porosity) * box_volume

    # selectable 是允许成为固相的面；硬尺寸模式会在拓扑搜索前直接剔除越界面。
    selectable = {
        face_id
        for face_id, face in enumerate(faces)
        if _area(face) > 0.0
    }
    if allowed_face_ids is not None:
        selectable.intersection_update(int(face_id) for face_id in allowed_face_ids)
    if hard_diameter_range:
        if diameter_range_nm is None:
            raise ValueError("hard_diameter_range requires diameter_range_nm")
        lower, upper = diameter_range_nm
        selectable = {
            face_id
            for face_id in selectable
            if lower <= _diameter(faces[face_id]) <= upper
        }

    if not selectable:
        raise ValueError("no faces satisfy the selection constraints")

    pore_roots = _sample_roots(pore_graph.nodes, max_pore_roots, rng)
    candidates: list[tuple[tuple[float, ...], DualBackbone]] = []

    def face_cost(item: Hashable) -> float:
        face_id = int(item)
        return max(float(face_volume[face_id]), 1e-12)

    def preferred(face_id: int) -> bool:
        if diameter_range_nm is None:
            return True
        lower, upper = diameter_range_nm
        return lower <= _diameter(faces[face_id]) <= upper

    def pore_face_cost(item: Hashable) -> float:
        """优先把小面或尺寸越界面留给孔相，减少对有效固相的占用。"""

        face_id = int(item)
        multiplier = 1.0 if preferred(face_id) else 0.05
        if face_id not in selectable:
            multiplier = 0.001
        return max(float(face_volume[face_id]) * multiplier, 1e-12)

    def solid_face_cost(item: Hashable) -> float:
        """固相骨架强烈优先选择用户指定片径范围内的面。"""

        face_id = int(item)
        base = max(float(face_volume[face_id]), 1e-12)
        if diameter_range_nm is None or preferred(face_id):
            return base
        lower, upper = diameter_range_nm
        diameter = _diameter(faces[face_id])
        width = max(upper - lower, 1e-12)
        distance = lower - diameter if diameter < lower else diameter - upper
        return base * (20.0 + 20.0 * distance / width)

    # 对多个孔相起点建立候选双骨架，最后统一按可达体积范围和骨架代价排序。
    for pore_root in pore_roots:
        pore = build_axis_backbone(
            pore_graph,
            roots=[pore_root],
            mode="edges",
            offset_bound=offset_bound,
            item_cost=pore_face_cost,
            try_axis_orders=False,
        )
        if pore is None:
            continue
        pore = _extend_pore_backbone_with_spanning_tree(
            pore_graph,
            pore,
            item_cost=pore_face_cost,
        )
        if not pore.winding.connected or not pore.winding.has_three_axis_component:
            continue
        try:
            protected = frozenset(int(item) for item in pore.items)
        except (TypeError, ValueError):
            continue

        allowed_solid = selectable.difference(protected)
        solid_roots = _sample_roots(allowed_solid, max_solid_roots, rng)
        solid = build_axis_backbone(
            face_graph,
            roots=solid_roots,
            mode="nodes",
            offset_bound=offset_bound,
            allowed_nodes=allowed_solid,
            item_cost=solid_face_cost,
            try_axis_orders=False,
        )
        if solid is None:
            continue

        solid_ids = frozenset(int(item) for item in solid.items)
        if solid_ids & protected:
            continue
        # 只允许沿固相骨架所在连通分量生长，后续每新增一面都不会产生悬空孤岛。
        growable = connected_component(
            face_graph,
            solid_ids,
            allowed_nodes=allowed_solid,
        )
        if not solid_ids.issubset(growable):
            continue

        nominal_min = float(face_volume[list(solid_ids)].sum())
        # 面积×厚度会重复统计交叠区，因此此处只是可达固相体积的解析上界。
        nominal_max = min(float(face_volume[list(growable)].sum()), box_volume)
        distance_to_interval = max(
            nominal_min - target_volume,
            target_volume - nominal_max,
            0.0,
        )
        score = (
            0.0 if distance_to_interval == 0.0 else 1.0,
            distance_to_interval / box_volume,
            nominal_min / box_volume,
            float(face_volume[list(protected)].sum()) / box_volume,
        )
        candidates.append(
            (
                score,
                DualBackbone(
                    pore=pore,
                    solid=solid,
                    protected_face_ids=protected,
                    solid_face_ids=solid_ids,
                    growable_face_ids=growable,
                    nominal_min_solid_fraction=nominal_min / box_volume,
                    nominal_max_solid_fraction_upper_bound=nominal_max / box_volume,
                ),
            )
        )

    if not candidates:
        qualifier = " under the hard diameter range" if hard_diameter_range else ""
        raise RuntimeError(
            "could not construct disjoint three-axis pore and solid backbones"
            + qualifier
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


class _UniformPointIndex:
    """对固定 QMC 样本点建立的轻量均匀网格索引。"""

    def __init__(self, points: np.ndarray, cube_size: float) -> None:
        self.points = points
        self.cube_size = float(cube_size)
        sample_axis = max(round(len(points) ** (1.0 / 3.0)), 1)
        self.bins_per_axis = int(np.clip(sample_axis // 2, 8, 96))
        self.bin_size = self.cube_size / self.bins_per_axis
        ijk = np.floor(points / self.bin_size).astype(np.int32)
        np.clip(ijk, 0, self.bins_per_axis - 1, out=ijk)
        codes = (
            (ijk[:, 0] * self.bins_per_axis + ijk[:, 1])
            * self.bins_per_axis
            + ijk[:, 2]
        )
        self.order = np.argsort(codes, kind="stable").astype(np.int32)
        self.sorted_codes = codes[self.order]

    def query(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        lower = np.maximum(np.asarray(lower, dtype=float), 0.0)
        upper = np.minimum(np.asarray(upper, dtype=float), self.cube_size)
        if np.any(upper < lower):
            return np.empty(0, dtype=np.int32)

        low_bin = np.floor(lower / self.bin_size).astype(int)
        high_probe = np.nextafter(upper, -np.inf)
        high_bin = np.floor(high_probe / self.bin_size).astype(int)
        np.clip(low_bin, 0, self.bins_per_axis - 1, out=low_bin)
        np.clip(high_bin, 0, self.bins_per_axis - 1, out=high_bin)

        chunks: list[np.ndarray] = []
        n = self.bins_per_axis
        for i, j, k in product(
            range(low_bin[0], high_bin[0] + 1),
            range(low_bin[1], high_bin[1] + 1),
            range(low_bin[2], high_bin[2] + 1),
        ):
            code = (i * n + j) * n + k
            start = int(np.searchsorted(self.sorted_codes, code, side="left"))
            stop = int(np.searchsorted(self.sorted_codes, code, side="right"))
            if stop > start:
                chunks.append(self.order[start:stop])

        if not chunks:
            return np.empty(0, dtype=np.int32)
        indices = np.concatenate(chunks)
        candidate_points = self.points[indices]
        keep = np.all(candidate_points >= lower, axis=1) & np.all(
            candidate_points <= upper, axis=1
        )
        return indices[keep]


class SobolCoverageEstimator:
    """Incremental union-volume estimator for periodically repeated face slabs.

    Each face mask is cached as sparse indices into one fixed scrambled Sobol
    point set.  Reusing the points is important: adding faces then produces a
    monotone solid-volume curve even though the estimator is approximate.

    每个逻辑面按指定厚度视为有限棱柱，cKDTree 只召回可能落入该棱柱的
    Sobol 点，再做精确的平面距离与凸多边形内部判定。所有面共用同一套乱序
    Sobol 点，并用布尔并集计算固相体积：这会自动去除片层交叠体积，且新增
    片层时估计的固相体积严格不减。
    """

    def __init__(
        self,
        faces: Sequence[FaceLike],
        *,
        cube_size_nm: float,
        thickness_nm: float,
        qmc_power: int = 20,
        seed: int = 0,
        geometric_tolerance_nm: float = 1e-9,
    ) -> None:
        if cube_size_nm <= 0 or thickness_nm <= 0:
            raise ValueError("cube_size_nm and thickness_nm must be positive")
        if not 4 <= qmc_power <= 24:
            raise ValueError("qmc_power must lie in [4, 24]")
        self.faces = faces
        self.cube_size_nm = float(cube_size_nm)
        self.thickness_nm = float(thickness_nm)
        self.box_volume_nm3 = self.cube_size_nm**3
        self.tolerance_nm = float(geometric_tolerance_nm)
        # 乱序 Sobol 序列比同数量独立随机点更均匀，可稳定估计高孔隙率模型的小固相体积。
        sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
        self.points = sampler.random_base2(qmc_power) * self.cube_size_nm
        self.sample_count = len(self.points)
        # boxsize 使距离查询自动使用最小镜像，无需显式复制 3×3×3 个样本点集。
        self._periodic_tree = cKDTree(
            self.points,
            boxsize=self.cube_size_nm,
            compact_nodes=True,
            balanced_tree=True,
        )
        self._cache: dict[int, np.ndarray] = {}

    @property
    def sample_volume_nm3(self) -> float:
        return self.box_volume_nm3 / self.sample_count

    def empty_occupancy(self) -> np.ndarray:
        return np.zeros(self.sample_count, dtype=np.bool_)

    @staticmethod
    def _clean_vertices(face: FaceLike) -> np.ndarray:
        vertices = _vertices(face)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("face vertices must be an (N, 3) array with N >= 3")
        if len(vertices) > 3 and np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        return vertices

    def _periodic_image_shifts(
        self, vertices: np.ndarray, normal: np.ndarray
    ) -> Iterable[tuple[int, int, int]]:
        extension = 0.5 * self.thickness_nm * np.abs(normal)
        lower = vertices.min(axis=0) - extension
        upper = vertices.max(axis=0) + extension
        ranges: list[range] = []
        for axis in range(3):
            first = int(ceil((-upper[axis]) / self.cube_size_nm))
            last = int(floor((self.cube_size_nm - lower[axis]) / self.cube_size_nm))
            ranges.append(range(first, last + 1))
        return product(*ranges)

    def _inside_prism(
        self,
        points: np.ndarray,
        vertices: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        # 先检查法向距离，对应总厚度 thickness_nm 的对称片层。
        relative = points - vertices[0]
        distance = relative @ normal
        within_thickness = (
            np.abs(distance)
            <= 0.5 * self.thickness_nm + self.tolerance_nm
        )
        if not np.any(within_thickness):
            return within_thickness

        # 再投影到中面，用凸多边形的同向半平面测试判定面内覆盖。
        projected = points - distance[:, None] * normal[None, :]
        edges = np.roll(vertices, -1, axis=0) - vertices
        polygon_vector = np.sum(
            np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0
        )
        orientation = float(np.dot(polygon_vector, normal))
        if abs(orientation) < 1e-15:
            raise ValueError("degenerate face polygon")
        orientation_sign = 1.0 if orientation > 0 else -1.0

        offsets = projected[:, None, :] - vertices[None, :, :]
        side = np.einsum(
            "mki,i->mk", np.cross(edges[None, :, :], offsets), normal
        )
        within_polygon = np.all(
            orientation_sign * side >= -self.tolerance_nm, axis=1
        )
        return within_thickness & within_polygon

    def face_indices(self, face_id: int) -> np.ndarray:
        """返回落入指定周期有限厚片层的 Sobol 点索引，并缓存结果。"""

        cached = self._cache.get(face_id)
        if cached is not None:
            return cached

        face = self.faces[face_id]
        vertices = self._clean_vertices(face)
        normal = np.asarray(face.normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            raise ValueError(f"face {face_id} has a zero normal")
        normal /= norm
        centre = np.mean(vertices, axis=0)
        polygon_radius = float(np.max(np.linalg.norm(vertices - centre, axis=1)))
        query_radius = float(np.hypot(polygon_radius, 0.5 * self.thickness_nm))
        # 先用外接球做宽松召回，避免对全部 Sobol 点逐面判定。
        candidates = np.asarray(
            self._periodic_tree.query_ball_point(
                np.mod(centre, self.cube_size_nm),
                query_radius + self.tolerance_nm,
            ),
            dtype=np.int64,
        )
        if len(candidates):
            # 将候选点按最小镜像展开到面中心所在坐标片，正确处理跨边界面。
            wrapped_centre = np.mod(centre, self.cube_size_nm)
            delta = self.points[candidates] - wrapped_centre
            delta -= self.cube_size_nm * np.rint(delta / self.cube_size_nm)
            unwrapped_points = centre + delta
            inside = self._inside_prism(unwrapped_points, vertices, normal)
            indices = np.sort(candidates[inside]).astype(np.int32, copy=False)
        else:
            indices = np.empty(0, dtype=np.int32)
        self._cache[face_id] = indices
        return indices

    def marginal_count(self, face_id: int, occupancy: np.ndarray) -> int:
        indices = self.face_indices(face_id)
        return int(np.count_nonzero(~occupancy[indices]))

    def add_face(self, face_id: int, occupancy: np.ndarray) -> int:
        indices = self.face_indices(face_id)
        new = indices[~occupancy[indices]]
        occupancy[new] = True
        return int(len(new))

    def occupancy(self, face_ids: Iterable[int]) -> np.ndarray:
        occupied = self.empty_occupancy()
        for face_id in face_ids:
            self.add_face(int(face_id), occupied)
        return occupied

    def union_count(self, face_ids: Iterable[int]) -> int:
        return int(np.count_nonzero(self.occupancy(face_ids)))

    def solid_fraction_from_count(self, count: int) -> float:
        return float(count) / self.sample_count

    def volume_nm3_from_count(self, count: int) -> float:
        return self.solid_fraction_from_count(count) * self.box_volume_nm3


@dataclass(frozen=True, slots=True)
class AttainableRange:
    min_solid_fraction: float
    max_solid_fraction: float
    min_porosity: float
    max_porosity: float
    nominal_max_solid_fraction_upper_bound: float
    estimator: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_face_ids: tuple[int, ...]
    protected_pore_face_ids: tuple[int, ...]
    status: str
    target_porosity: float
    achieved_porosity: float
    absolute_porosity_error: float
    tolerance: float
    porosity_before_crossing: float | None
    porosity_after_crossing: float | None
    attainable_range: AttainableRange
    solid_winding: WindingReport
    pore_winding: WindingReport
    diameter_statistics_nm: dict[str, float]

    @property
    def within_tolerance(self) -> bool:
        return self.absolute_porosity_error <= self.tolerance


def _diameter_statistics(
    faces: Sequence[FaceLike], selected: Iterable[int]
) -> dict[str, float]:
    face_ids = list(selected)
    if not face_ids:
        return {}
    diameters = np.asarray([_diameter(faces[index]) for index in face_ids])
    areas = np.asarray([_area(faces[index]) for index in face_ids])
    result = {
        "min": float(diameters.min()),
        "q10": float(np.quantile(diameters, 0.10)),
        "median": float(np.median(diameters)),
        "q90": float(np.quantile(diameters, 0.90)),
        "max": float(diameters.max()),
        "mean": float(diameters.mean()),
    }
    if areas.sum() > 0:
        result["area_weighted_mean"] = float(np.average(diameters, weights=areas))
    return result


def _open_pore_edge_ids(
    pore_graph: PeriodicGraph, selected_face_ids: set[int]
) -> set[int]:
    open_edges = set()
    for edge in pore_graph.iter_edges():
        if edge.payload is None or int(edge.payload) not in selected_face_ids:
            open_edges.add(edge.edge_id)
    return open_edges


def select_by_connected_frontier(
    faces: Sequence[FaceLike],
    face_graph: PeriodicGraph,
    backbone: DualBackbone,
    estimator: SobolCoverageEstimator,
    *,
    target_porosity: float,
    tolerance: float = 0.005,
    random_seed: int = 0,
    preferred_diameter_range_nm: tuple[float, float] | None = None,
    min_preferred_fraction: float = 0.0,
    max_selected_faces: int | None = None,
    evaluate_maximum: bool = True,
    near_target_candidate_limit: int = 256,
) -> SelectionResult:
    """Grow complete logical faces from the solid backbone to target porosity.

    从已三向贯通的固相骨架出发，每次只在其连接前沿添加一个完整逻辑面。
    被孔相生成树保护的面永不参与选择，因此生长过程中固相不会断开，
    孔相也不会被分割成孤立闭孔。
    """

    if not 0.0 < target_porosity < 1.0:
        raise ValueError("target_porosity must lie strictly between zero and one")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not 0.0 <= min_preferred_fraction <= 1.0:
        raise ValueError("min_preferred_fraction must lie in [0, 1]")

    pore_graph = build_pore_graph(faces)
    selected = set(backbone.solid_face_ids)
    protected = set(backbone.protected_face_ids)
    growable = set(backbone.growable_face_ids)
    if selected & protected:
        raise ValueError("solid and pore backbones overlap")

    # 最稀疏的可行状态就是固相骨架本身；其 QMC 覆盖数给出最大可达孔隙率。
    occupancy = estimator.occupancy(selected)
    current_count = int(np.count_nonzero(occupancy))
    min_count = current_count
    if evaluate_maximum:
        # 将整个可生长连通分量取并集，得到最致密可行状态及最小可达孔隙率。
        max_count = estimator.union_count(growable)
        range_estimator = f"Sobol QMC 2^{int(np.log2(estimator.sample_count))}"
    else:
        nominal = min(
            backbone.nominal_max_solid_fraction_upper_bound,
            1.0,
        )
        max_count = max(min_count, int(round(nominal * estimator.sample_count)))
        range_estimator = "minimum QMC; maximum area-thickness upper bound"

    min_solid = estimator.solid_fraction_from_count(min_count)
    max_solid = estimator.solid_fraction_from_count(max_count)
    attainable = AttainableRange(
        min_solid_fraction=min_solid,
        max_solid_fraction=max_solid,
        min_porosity=1.0 - max_solid,
        max_porosity=1.0 - min_solid,
        nominal_max_solid_fraction_upper_bound=(
            backbone.nominal_max_solid_fraction_upper_bound
        ),
        estimator=range_estimator,
    )

    target_count = int(round((1.0 - target_porosity) * estimator.sample_count))

    def make_result(
        chosen: set[int],
        count: int,
        status: str,
        before: float | None = None,
        after: float | None = None,
    ) -> SelectionResult:
        solid_winding = analyze_winding(face_graph, active_nodes=chosen)
        open_edges = _open_pore_edge_ids(pore_graph, chosen)
        pore_winding = analyze_winding(pore_graph, active_edge_ids=open_edges)
        porosity = 1.0 - estimator.solid_fraction_from_count(count)
        if not solid_winding.connected or not solid_winding.has_three_axis_component:
            status = "TOPOLOGY_FAILURE_SOLID"
        elif not pore_winding.connected or not pore_winding.has_three_axis_component:
            status = "TOPOLOGY_FAILURE_PORE"
        diameter_statistics = _diameter_statistics(faces, chosen)
        if preferred_diameter_range_nm is not None and chosen:
            lower, upper = preferred_diameter_range_nm
            in_range = sum(
                lower <= _diameter(faces[face_id]) <= upper
                for face_id in chosen
            )
            diameter_statistics["in_preferred_range_fraction"] = (
                float(in_range) / len(chosen)
            )
            if (
                diameter_statistics["in_preferred_range_fraction"]
                + 1e-12
                < min_preferred_fraction
                and not status.startswith("TOPOLOGY_FAILURE")
            ):
                status = "SIZE_RANGE_FRACTION_FAILURE"
        return SelectionResult(
            selected_face_ids=tuple(sorted(chosen)),
            protected_pore_face_ids=tuple(sorted(protected)),
            status=status,
            target_porosity=target_porosity,
            achieved_porosity=porosity,
            absolute_porosity_error=abs(porosity - target_porosity),
            tolerance=tolerance,
            porosity_before_crossing=before,
            porosity_after_crossing=after,
            attainable_range=attainable,
            solid_winding=solid_winding,
            pore_winding=pore_winding,
            diameter_statistics_nm=diameter_statistics,
        )

    # 在进入随机生长前先检查目标是否落在双骨架约束下的可达区间。
    if target_porosity > attainable.max_porosity + tolerance:
        return make_result(selected, current_count, "INFEASIBLE_TOO_POROUS")
    if target_porosity < attainable.min_porosity - tolerance:
        max_state = set(growable)
        return make_result(max_state, max_count, "INFEASIBLE_TOO_DENSE")
    if current_count >= target_count:
        status = (
            "REACHED" if abs((1.0 - min_solid) - target_porosity) <= tolerance
            else "BRACKETED_DISCRETE"
        )
        return make_result(selected, current_count, status)

    rng = np.random.default_rng(random_seed)
    priorities: dict[int, float] = {}
    frontier: set[int] = set()
    queue: list[tuple[float, int]] = []

    def priority(face_id: int) -> float:
        """为前沿面生成可复现的随机优先级，并对片径越界面施加大惩罚。"""

        value = priorities.get(face_id)
        if value is not None:
            return value
        penalty = 0.0
        if preferred_diameter_range_nm is not None:
            lower, upper = preferred_diameter_range_nm
            diameter = _diameter(faces[face_id])
            width = max(upper - lower, 1e-12)
            if diameter < lower:
                penalty = 1_000_000.0 + (lower - diameter) / width
            elif diameter > upper:
                penalty = 1_000_000.0 + (diameter - upper) / width
        value = float(rng.exponential()) + penalty
        priorities[face_id] = value
        return value

    def expose(face_id: int) -> None:
        """把新选面的未选相邻面暴露为生长前沿，同时排除孔相保护面。"""

        for arc in face_graph.arcs_from(face_id):
            neighbor = int(arc.target)
            if (
                neighbor in growable
                and neighbor not in selected
                and neighbor not in protected
                and neighbor not in frontier
            ):
                frontier.add(neighbor)
                heappush(queue, (priority(neighbor), neighbor))

    for face_id in tuple(selected):
        expose(face_id)

    nominal_sample_counts = np.asarray(
        [
            _area(face) * estimator.thickness_nm
            / estimator.box_volume_nm3
            * estimator.sample_count
            for face in faces
        ]
    )
    typical_step = max(float(np.median(nominal_sample_counts)), 1.0)
    before_selected: set[int] | None = None
    before_count: int | None = None
    after_selected: set[int] | None = None
    after_count: int | None = None

    # 只从当前前沿选面，这是固相在整个生长过程中始终连通的直接保证。
    while current_count < target_count and frontier:
        if max_selected_faces is not None and len(selected) >= max_selected_faces:
            break
        remaining = target_count - current_count

        if remaining <= 8.0 * typical_step:
            # 接近目标时显式比较多个候选面的真实边际并集体积，降低单面离散步长造成的超调。
            candidate_pool = sorted(frontier, key=priority)[
                :near_target_candidate_limit
            ]
            marginal = {
                face_id: estimator.marginal_count(face_id, occupancy)
                for face_id in candidate_pool
            }
            positive = [face_id for face_id in candidate_pool if marginal[face_id] > 0]
            if positive:
                chosen = min(
                    positive,
                    key=lambda face_id: (
                        abs(remaining - marginal[face_id]),
                        priority(face_id),
                    ),
                )
            else:
                chosen = candidate_pool[0]
        else:
            chosen = -1
            while queue:
                _, candidate = heappop(queue)
                if candidate in frontier:
                    chosen = candidate
                    break
            if chosen < 0:
                break

        # 保留跨过目标前的完整状态，以便稍后与跨过后状态比较。
        before_selected = set(selected)
        before_count = current_count
        frontier.discard(chosen)
        selected.add(chosen)
        current_count += estimator.add_face(chosen, occupancy)
        expose(chosen)

        if current_count >= target_count:
            after_selected = set(selected)
            after_count = current_count
            break

    if before_selected is None or before_count is None:
        return make_result(selected, current_count, "FRONTIER_EXHAUSTED")

    before_porosity = 1.0 - estimator.solid_fraction_from_count(before_count)
    if after_selected is None or after_count is None:
        achieved_porosity = 1.0 - estimator.solid_fraction_from_count(current_count)
        status = (
            "REACHED" if abs(achieved_porosity - target_porosity) <= tolerance
            else "FRONTIER_EXHAUSTED"
        )
        return make_result(
            selected,
            current_count,
            status,
            before_porosity,
            achieved_porosity,
        )

    after_porosity = 1.0 - estimator.solid_fraction_from_count(after_count)
    # 单个逻辑面不能拆分，目标通常夹在两个离散状态之间；选孔隙率误差更小的一侧。
    if abs(before_porosity - target_porosity) <= abs(
        after_porosity - target_porosity
    ):
        chosen_state, chosen_count = before_selected, before_count
    else:
        chosen_state, chosen_count = after_selected, after_count

    chosen_porosity = 1.0 - estimator.solid_fraction_from_count(chosen_count)
    status = (
        "REACHED"
        if abs(chosen_porosity - target_porosity) <= tolerance
        else "BRACKETED_DISCRETE"
    )
    return make_result(
        chosen_state,
        chosen_count,
        status,
        before_porosity,
        after_porosity,
    )


def plan_and_select(
    faces: Sequence[FaceLike],
    face_adjacency: PeriodicGraph
    | Mapping[Hashable, Iterable[object]]
    | Sequence[Iterable[object]],
    *,
    cube_size_nm: float,
    thickness_nm: float,
    target_porosity: float,
    tolerance: float = 0.005,
    diameter_range_nm: tuple[float, float] | None = None,
    hard_diameter_range: bool = False,
    allowed_face_ids: Iterable[int] | None = None,
    min_preferred_fraction: float = 0.0,
    qmc_power: int = 20,
    random_seed: int = 0,
    offset_bound: int = 1,
    max_selected_faces: int | None = None,
) -> tuple[SelectionResult, DualBackbone, SobolCoverageEstimator]:
    """Convenience wrapper joining graph, backbone, QMC, and growth stages.

    便捷入口串联四个阶段：面邻接图标准化、固/孔双骨架构建、Sobol 有限厚
    片层并集体积估计，以及从连接前沿生长到目标孔隙率。
    """

    face_graph = coerce_face_graph(face_adjacency, face_count=len(faces))
    backbone = construct_dual_backbone(
        faces,
        face_graph,
        cube_size_nm=cube_size_nm,
        thickness_nm=thickness_nm,
        target_porosity=target_porosity,
        diameter_range_nm=diameter_range_nm,
        hard_diameter_range=hard_diameter_range,
        allowed_face_ids=allowed_face_ids,
        offset_bound=offset_bound,
        random_seed=random_seed,
    )
    estimator = SobolCoverageEstimator(
        faces,
        cube_size_nm=cube_size_nm,
        thickness_nm=thickness_nm,
        qmc_power=qmc_power,
        seed=random_seed,
    )
    result = select_by_connected_frontier(
        faces,
        face_graph,
        backbone,
        estimator,
        target_porosity=target_porosity,
        tolerance=tolerance,
        random_seed=random_seed,
        preferred_diameter_range_nm=diameter_range_nm,
        min_preferred_fraction=min_preferred_fraction,
        max_selected_faces=max_selected_faces,
    )
    return result, backbone, estimator


__all__ = [
    "AttainableRange",
    "DualBackbone",
    "SelectionResult",
    "SobolCoverageEstimator",
    "build_pore_graph",
    "coerce_face_graph",
    "construct_dual_backbone",
    "plan_and_select",
    "select_by_connected_frontier",
]
