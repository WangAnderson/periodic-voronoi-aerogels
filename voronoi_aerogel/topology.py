"""Periodic graph utilities for bicontinuous Voronoi sheet networks.

The geometry layer is intentionally not imported here.  A periodic graph edge
stores an integer lattice displacement: traversing ``u -> v`` moves from a
copy of ``u`` in lattice cell ``k`` to a copy of ``v`` in ``k + shift``.
This small abstraction is sufficient for connectivity and winding tests for
both the pore-cell graph and the selected-face graph.

这里只处理周期拓扑，不依赖具体几何。图边的整数 ``shift`` 表示沿该边行走时
跨越了多少个周期立方体，同一套算法因而可同时检验固相片层网络与孔相网络。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import permutations
from math import inf
from typing import Callable, Hashable, Iterable, Iterator, Mapping, Sequence

import numpy as np


Node = Hashable
Shift = tuple[int, int, int]


def _shift(value: Sequence[int] | np.ndarray) -> Shift:
    if len(value) != 3:
        raise ValueError("periodic shifts must contain exactly three integers")
    return (int(value[0]), int(value[1]), int(value[2]))


def add_shift(a: Shift, b: Shift) -> Shift:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_shift(a: Shift, b: Shift) -> Shift:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def neg_shift(a: Shift) -> Shift:
    return (-a[0], -a[1], -a[2])


@dataclass(frozen=True, slots=True)
class PeriodicArc:
    """One directed view of an undirected periodic graph edge.

    无向周期边的一个有向视图；反向弧共用 ``edge_id`` 且 ``shift`` 取反。
    """

    edge_id: int
    source: Node
    target: Node
    shift: Shift
    payload: Hashable | None = None


class PeriodicGraph:
    """An undirected quotient graph with integer translations on its edges.

    节点只存主单元中的等价类，周期副本不另建节点；副本信息由边上的
    整数格矢量保留，这就是无限周期图的商图表示。
    """

    def __init__(self, nodes: Iterable[Node] = ()) -> None:
        self._adjacency: dict[Node, list[PeriodicArc]] = {
            node: [] for node in nodes
        }
        self._edges: dict[int, PeriodicArc] = {}
        self._next_edge_id = 0

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._adjacency)

    @property
    def edge_ids(self) -> frozenset[int]:
        return frozenset(self._edges)

    def __len__(self) -> int:
        return len(self._adjacency)

    def add_node(self, node: Node) -> None:
        self._adjacency.setdefault(node, [])

    def add_edge(
        self,
        source: Node,
        target: Node,
        shift: Sequence[int] | np.ndarray = (0, 0, 0),
        *,
        payload: Hashable | None = None,
        edge_id: int | None = None,
    ) -> int:
        """Add an undirected edge and return its shared edge identifier.

        内部同时存储正反两条弧，保证任意方向遍历时的周期位移互为相反数。
        """

        lattice_shift = _shift(shift)
        if edge_id is None:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
        elif edge_id in self._edges:
            raise ValueError(f"edge_id {edge_id} is already present")
        else:
            self._next_edge_id = max(self._next_edge_id, edge_id + 1)

        self.add_node(source)
        self.add_node(target)
        forward = PeriodicArc(
            edge_id, source, target, lattice_shift, payload
        )
        reverse = PeriodicArc(
            edge_id, target, source, neg_shift(lattice_shift), payload
        )
        self._adjacency[source].append(forward)
        self._adjacency[target].append(reverse)
        self._edges[edge_id] = forward
        return edge_id

    def arcs_from(self, node: Node) -> tuple[PeriodicArc, ...]:
        return tuple(self._adjacency.get(node, ()))

    def edge(self, edge_id: int) -> PeriodicArc:
        return self._edges[edge_id]

    def iter_edges(self) -> Iterator[PeriodicArc]:
        return iter(self._edges.values())

    def copy(self) -> "PeriodicGraph":
        copied = PeriodicGraph(self.nodes)
        for edge in self.iter_edges():
            copied.add_edge(
                edge.source,
                edge.target,
                edge.shift,
                payload=edge.payload,
                edge_id=edge.edge_id,
            )
        return copied


def graph_from_edges(
    edges: Iterable[
        PeriodicArc
        | tuple[Node, Node, Sequence[int]]
        | tuple[Node, Node, Sequence[int], Hashable | None]
    ],
    *,
    nodes: Iterable[Node] = (),
) -> PeriodicGraph:
    """Build a graph from a deliberately small, geometry-agnostic format."""

    graph = PeriodicGraph(nodes)
    for item in edges:
        if isinstance(item, PeriodicArc):
            graph.add_edge(
                item.source,
                item.target,
                item.shift,
                payload=item.payload,
            )
        elif len(item) == 3:
            source, target, shift = item
            graph.add_edge(source, target, shift)
        elif len(item) == 4:
            source, target, shift, payload = item
            graph.add_edge(source, target, shift, payload=payload)
        else:
            raise ValueError("edges must have 3 or 4 fields")
    return graph


def graph_from_adjacency(
    adjacency: Mapping[Node, Iterable[object]] | Sequence[Iterable[object]],
) -> PeriodicGraph:
    """Convert common adjacency-list layouts into :class:`PeriodicGraph`.

    Each adjacency item may be ``neighbor``, ``(neighbor, shift)``, or
    ``(neighbor, shift, payload)``.  Because an adjacency list normally stores
    both directions, duplicate reverse arcs are removed.  Plain ``neighbor``
    entries are interpreted as zero-shift edges.
    """

    items = adjacency.items() if isinstance(adjacency, Mapping) else enumerate(adjacency)
    graph = PeriodicGraph(node for node, _ in items)
    items = adjacency.items() if isinstance(adjacency, Mapping) else enumerate(adjacency)
    seen: set[tuple[object, ...]] = set()

    for source, neighbors in items:
        for raw in neighbors:
            if isinstance(raw, PeriodicArc):
                target, shift, payload = raw.target, raw.shift, raw.payload
            elif isinstance(raw, tuple) and len(raw) == 3:
                target, shift, payload = raw
            elif isinstance(raw, tuple) and len(raw) == 2:
                target, shift = raw
                payload = None
            else:
                target, shift, payload = raw, (0, 0, 0), None

            shift = _shift(shift)
            forward = (source, target, shift, payload)
            reverse = (target, source, neg_shift(shift), payload)
            key = min((forward, reverse), key=repr)
            if key in seen:
                continue
            seen.add(key)
            graph.add_edge(source, target, shift, payload=payload)
    return graph


@dataclass(frozen=True, slots=True)
class WindingComponent:
    """一个连通分量的周期绕行信息。

    ``winding_vectors`` 是不可缩周期回路的整数格位移，``rank`` 是其线性无关基的维数。
    本项目要求 rank=3，才能说明网络在三个独立方向上均周期贯通。
    """

    nodes: frozenset[Node]
    rank: int
    basis: tuple[Shift, ...]
    axis_mask: tuple[bool, bool, bool]
    winding_vectors: tuple[Shift, ...]

    @property
    def winds_all_axes(self) -> bool:
        return all(self.axis_mask)

    @property
    def full_rank(self) -> bool:
        return self.rank == 3


@dataclass(frozen=True, slots=True)
class WindingReport:
    """周期图各连通分量的贯通性汇总。"""

    components: tuple[WindingComponent, ...]

    @property
    def connected(self) -> bool:
        return len(self.components) <= 1

    @property
    def maximum_rank(self) -> int:
        return max((component.rank for component in self.components), default=0)

    @property
    def has_full_rank_component(self) -> bool:
        return any(component.full_rank for component in self.components)

    @property
    def has_three_axis_component(self) -> bool:
        return any(
            component.full_rank and component.winds_all_axes
            for component in self.components
        )


def _independent_basis(vectors: Iterable[Shift]) -> tuple[Shift, ...]:
    """Return up to three linearly independent integer vectors.

    仅某个绕行矢量的 x/y/z 分量都非零并不代表三向独立贯通；
    例如 ``(1, 1, 1)`` 的秩仍然只是 1，因此必须显式计算线性无关基。
    """

    basis: list[Shift] = []
    rank = 0
    for vector in vectors:
        if vector == (0, 0, 0):
            continue
        candidate = basis + [vector]
        matrix = np.asarray(candidate, dtype=np.float64)
        new_rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
        if new_rank > rank:
            basis.append(vector)
            rank = new_rank
            if rank == 3:
                break
    return tuple(basis)


def analyze_winding(
    graph: PeriodicGraph,
    *,
    active_nodes: Iterable[Node] | None = None,
    active_edge_ids: Iterable[int] | None = None,
) -> WindingReport:
    """Find connected components and non-contractible cycle rank.

    Potentials are integer lattice coordinates assigned during a graph walk.
    Any inconsistent already-visited edge closes a cycle; its residual is the
    cycle winding vector.  Rank is evaluated in R^3, while the raw integer
    vectors are retained for stricter downstream checks.

    遍历时给每个节点分配一个展开空间整数坐标。若从另一条路径到达已访问
    节点时得到不同坐标，两坐标之差就是该闭合路径穿越周期边界的绕行矢量。
    只有存在秩为 3 的连通分量，才视为三方向周期贯通。
    """

    nodes = set(graph.nodes if active_nodes is None else active_nodes)
    edge_ids = None if active_edge_ids is None else set(active_edge_ids)
    unseen = set(nodes)
    components: list[WindingComponent] = []

    while unseen:
        root = unseen.pop()
        # potential[node] 是该节点在当前遍历树中所处的整数周期层。
        potential: dict[Node, Shift] = {root: (0, 0, 0)}
        stack = [root]
        component_nodes = {root}
        windings: set[Shift] = set()

        while stack:
            source = stack.pop()
            for arc in graph.arcs_from(source):
                if arc.target not in nodes:
                    continue
                if edge_ids is not None and arc.edge_id not in edge_ids:
                    continue

                expected = add_shift(potential[source], arc.shift)
                if arc.target not in potential:
                    potential[arc.target] = expected
                    component_nodes.add(arc.target)
                    unseen.discard(arc.target)
                    stack.append(arc.target)
                    continue

                winding = sub_shift(expected, potential[arc.target])
                if winding != (0, 0, 0):
                    # 非零残差表明此边闭合了一条不可缩的周期回路。
                    canonical = min(winding, neg_shift(winding))
                    windings.add(canonical)

        ordered_windings = tuple(sorted(windings))
        basis = _independent_basis(ordered_windings)
        axis_mask = tuple(
            any(vector[axis] != 0 for vector in ordered_windings)
            for axis in range(3)
        )
        components.append(
            WindingComponent(
                nodes=frozenset(component_nodes),
                rank=len(basis),
                basis=basis,
                axis_mask=axis_mask,  # type: ignore[arg-type]
                winding_vectors=ordered_windings,
            )
        )

    components.sort(key=lambda component: len(component.nodes), reverse=True)
    return WindingReport(tuple(components))


@dataclass(frozen=True, slots=True)
class LiftedPath:
    """商图路径在周期覆盖空间中的完整记录。

    ``offsets`` 记录每个路径节点所在的整数周期层，因而能区分同一商节点的不同副本。
    """

    nodes: tuple[Node, ...]
    offsets: tuple[Shift, ...]
    edge_ids: tuple[int, ...]
    payloads: tuple[Hashable | None, ...]
    cost: float


def shortest_lifted_path(
    graph: PeriodicGraph,
    start: Node,
    target_shift: Sequence[int],
    *,
    offset_bound: int = 1,
    allowed_nodes: Iterable[Node] | None = None,
    allowed_edge_ids: Iterable[int] | None = None,
    edge_cost: Callable[[PeriodicArc], float] | None = None,
    node_cost: Callable[[Node], float] | None = None,
) -> LiftedPath | None:
    """Shortest path from ``(start, 0)`` to a translated copy of ``start``.

    Dijkstra 的状态是 ``(商节点, 整数周期层)``，而不只是商节点。从起点走到
    它自身的 ``target_shift`` 副本，就给出一条具有指定绕行的显式周期通路。
    ``offset_bound`` 将无限覆盖空间截断为有限搜索区域。
    """

    if offset_bound < 1:
        raise ValueError("offset_bound must be at least one")
    target = _shift(target_shift)
    if any(abs(value) > offset_bound for value in target):
        raise ValueError("target_shift lies outside offset_bound")

    nodes = set(graph.nodes if allowed_nodes is None else allowed_nodes)
    if start not in nodes:
        return None
    edge_ids = None if allowed_edge_ids is None else set(allowed_edge_ids)
    edge_cost = edge_cost or (lambda _arc: 1.0)
    node_cost = node_cost or (lambda _node: 0.0)

    start_state = (start, (0, 0, 0))
    goal_state = (start, target)
    distances: dict[tuple[Node, Shift], float] = {start_state: 0.0}
    previous: dict[
        tuple[Node, Shift], tuple[tuple[Node, Shift], PeriodicArc]
    ] = {}
    queue: list[tuple[float, int, tuple[Node, Shift]]] = []
    sequence = 0
    heappush(queue, (0.0, sequence, start_state))

    while queue:
        distance, _, state = heappop(queue)
        if distance != distances.get(state):
            continue
        if state == goal_state:
            break

        source, offset = state
        for arc in graph.arcs_from(source):
            if arc.target not in nodes:
                continue
            if edge_ids is not None and arc.edge_id not in edge_ids:
                continue
            next_offset = add_shift(offset, arc.shift)
            # 同一商节点在不同 offset 上是不同的覆盖空间状态。
            if any(abs(value) > offset_bound for value in next_offset):
                continue

            step_cost = float(edge_cost(arc)) + float(node_cost(arc.target))
            if step_cost < 0 or not np.isfinite(step_cost):
                continue
            next_state = (arc.target, next_offset)
            next_distance = distance + step_cost
            if next_distance + 1e-15 >= distances.get(next_state, inf):
                continue

            distances[next_state] = next_distance
            previous[next_state] = (state, arc)
            sequence += 1
            heappush(queue, (next_distance, sequence, next_state))

    if goal_state not in distances:
        return None

    states = [goal_state]
    arcs: list[PeriodicArc] = []
    cursor = goal_state
    while cursor != start_state:
        cursor, arc = previous[cursor]
        states.append(cursor)
        arcs.append(arc)
    states.reverse()
    arcs.reverse()

    return LiftedPath(
        nodes=tuple(state[0] for state in states),
        offsets=tuple(state[1] for state in states),
        edge_ids=tuple(arc.edge_id for arc in arcs),
        payloads=tuple(arc.payload for arc in arcs),
        cost=distances[goal_state],
    )


@dataclass(frozen=True, slots=True)
class AxisBackbone:
    """A connected set containing explicit unit-winding paths.

    骨架中保存从同一根节点到 x/y/z 单位周期副本的显式路径，
    用于在后续随机选面之前先锁定一个连通且三向贯通的最小网络。
    """

    root: Node
    mode: str
    nodes: frozenset[Node]
    edge_ids: frozenset[int]
    items: frozenset[Hashable]
    paths: tuple[LiftedPath, ...]
    cost: float
    winding: WindingReport


def build_axis_backbone(
    graph: PeriodicGraph,
    *,
    roots: Iterable[Node],
    mode: str,
    axes: Sequence[int] = (0, 1, 2),
    offset_bound: int = 1,
    allowed_nodes: Iterable[Node] | None = None,
    allowed_edge_ids: Iterable[int] | None = None,
    item_cost: Callable[[Hashable], float] | None = None,
    try_axis_orders: bool = True,
) -> AxisBackbone | None:
    """Construct a low-cost connected backbone with unit winding paths.

    ``mode='nodes'`` selects graph nodes (solid Voronoi faces).
    ``mode='edges'`` selects edge payloads (open pore throats).  If an edge has
    no payload, its integer edge identifier is used as the selected item.

    ``mode='nodes'`` 用于以 Voronoi 面为节点的固相图；``mode='edges'`` 用于以
    未封闭公共面为边的孔相图。各方向路径允许共用已选元素，所以返回的是较低代价
    而且始终连到同一根的贯通骨架。
    """

    if mode not in {"nodes", "edges"}:
        raise ValueError("mode must be 'nodes' or 'edges'")
    axis_tuple = tuple(int(axis) for axis in axes)
    if any(axis not in (0, 1, 2) for axis in axis_tuple):
        raise ValueError("axes must contain only 0, 1, and 2")

    allowed_node_set = set(graph.nodes if allowed_nodes is None else allowed_nodes)
    allowed_edge_set = None if allowed_edge_ids is None else set(allowed_edge_ids)
    item_cost = item_cost or (lambda _item: 1.0)
    orders = (
        tuple(permutations(axis_tuple))
        if try_axis_orders and len(axis_tuple) > 1
        else (axis_tuple,)
    )

    best: AxisBackbone | None = None
    for root in roots:
        if root not in allowed_node_set:
            continue
        for order in orders:
            selected_nodes: set[Node] = {root} if mode == "nodes" else set()
            selected_edges: set[int] = set()
            selected_items: set[Hashable] = {root} if mode == "nodes" else set()
            paths: list[LiftedPath] = []
            failed = False

            for axis in order:
                target = [0, 0, 0]
                target[axis] = 1

                if mode == "nodes":
                    def node_weight(node: Node) -> float:
                        # 已纳入骨架的节点代价置零，鼓励三条路径共用结构。
                        return 0.0 if node in selected_nodes else item_cost(node)

                    path = shortest_lifted_path(
                        graph,
                        root,
                        target,
                        offset_bound=offset_bound,
                        allowed_nodes=allowed_node_set,
                        allowed_edge_ids=allowed_edge_set,
                        edge_cost=lambda _arc: 0.0,
                        node_cost=node_weight,
                    )
                else:
                    def edge_weight(arc: PeriodicArc) -> float:
                        item = arc.payload if arc.payload is not None else arc.edge_id
                        # 边模式下同理优先复用已保护的孔道。
                        return 0.0 if item in selected_items else item_cost(item)

                    path = shortest_lifted_path(
                        graph,
                        root,
                        target,
                        offset_bound=offset_bound,
                        allowed_nodes=allowed_node_set,
                        allowed_edge_ids=allowed_edge_set,
                        edge_cost=edge_weight,
                        node_cost=lambda _node: 0.0,
                    )

                if path is None:
                    failed = True
                    break
                paths.append(path)
                selected_nodes.update(path.nodes)
                selected_edges.update(path.edge_ids)
                if mode == "nodes":
                    selected_items.update(path.nodes)
                else:
                    selected_items.update(
                        payload if payload is not None else edge_id
                        for payload, edge_id in zip(path.payloads, path.edge_ids)
                    )

            if failed:
                continue

            active_nodes = selected_nodes if mode == "nodes" else None
            active_edges = selected_edges if mode == "edges" else None
            winding = analyze_winding(
                graph,
                active_nodes=active_nodes,
                active_edge_ids=active_edges,
            )
            if not winding.has_three_axis_component and set(axis_tuple) == {0, 1, 2}:
                continue

            cost = sum(float(item_cost(item)) for item in selected_items)
            candidate = AxisBackbone(
                root=root,
                mode=mode,
                nodes=frozenset(selected_nodes),
                edge_ids=frozenset(selected_edges),
                items=frozenset(selected_items),
                paths=tuple(paths),
                cost=cost,
                winding=winding,
            )
            if best is None or candidate.cost < best.cost:
                best = candidate

    return best


def connected_component(
    graph: PeriodicGraph,
    roots: Iterable[Node],
    *,
    allowed_nodes: Iterable[Node] | None = None,
    allowed_edge_ids: Iterable[int] | None = None,
) -> frozenset[Node]:
    """Return the union of components reached from ``roots``."""

    allowed = set(graph.nodes if allowed_nodes is None else allowed_nodes)
    edge_ids = None if allowed_edge_ids is None else set(allowed_edge_ids)
    reached = {root for root in roots if root in allowed}
    stack = list(reached)
    while stack:
        node = stack.pop()
        for arc in graph.arcs_from(node):
            if arc.target not in allowed or arc.target in reached:
                continue
            if edge_ids is not None and arc.edge_id not in edge_ids:
                continue
            reached.add(arc.target)
            stack.append(arc.target)
    return frozenset(reached)


def validate_reverse_arcs(graph: PeriodicGraph) -> None:
    """Raise if an edge does not have the expected reverse representation."""

    for edge in graph.iter_edges():
        matches = [
            arc
            for arc in graph.arcs_from(edge.target)
            if (
                arc.edge_id == edge.edge_id
                and arc.target == edge.source
                and arc.shift == neg_shift(edge.shift)
            )
        ]
        # A self-loop stores both directed views in the same adjacency list.
        # For a zero-shift self-loop the views are identical, so uniqueness is
        # not a meaningful invariant; existence is sufficient.
        if not matches:
            raise ValueError(f"edge {edge.edge_id} has an inconsistent reverse arc")


__all__ = [
    "AxisBackbone",
    "LiftedPath",
    "PeriodicArc",
    "PeriodicGraph",
    "Shift",
    "WindingComponent",
    "WindingReport",
    "add_shift",
    "analyze_winding",
    "build_axis_backbone",
    "connected_component",
    "graph_from_adjacency",
    "graph_from_edges",
    "neg_shift",
    "shortest_lifted_path",
    "sub_shift",
    "validate_reverse_arcs",
]
