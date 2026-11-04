"""Connected edge selection and periodic nanorod volume estimation.

Logical Voronoi edges are treated as capsules: a cylindrical centre section
with spherical end caps.  Every edge is selected as a whole before its
periodic images are clipped by the representative cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor, pi
from numbers import Integral
from typing import Hashable, Iterable, Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from voronoi_aerogel.topology import (
    PeriodicGraph,
    WindingReport,
    analyze_winding,
    build_axis_backbone,
)


class EdgeLike(Protocol):
    """Geometry fields required by this module."""

    id: int
    start_vertex: int
    end_vertex: int
    shift: object
    start_nm: object
    end_nm: object
    length_nm: float
    incident_face_ids: object


class CapsuleCoverageEstimator:
    """Sobol estimate of the union volume of periodic capsule-shaped rods.

    A single fixed point set is reused for every edge.  Each capsule therefore
    has a cached sparse list of covered sample indices, and union estimates are
    monotone while edges are added.  Periodicity is handled exactly by testing
    every translated capsule image whose bounding box intersects the primary
    cube.
    """

    def __init__(
        self,
        edges: Sequence[EdgeLike],
        *,
        cube_size_nm: float,
        rod_diameter_nm: float,
        qmc_power: int = 19,
        seed: int = 0,
        geometric_tolerance_nm: float | None = None,
    ) -> None:
        if cube_size_nm <= 0.0:
            raise ValueError("cube_size_nm must be positive")
        if rod_diameter_nm <= 0.0:
            raise ValueError("rod_diameter_nm must be positive")
        if not 4 <= qmc_power <= 24:
            raise ValueError("qmc_power must lie in [4, 24]")

        self.edges = tuple(edges)
        self.cube_size_nm = float(cube_size_nm)
        self.rod_diameter_nm = float(rod_diameter_nm)
        self.radius_nm = 0.5 * self.rod_diameter_nm
        self.box_volume_nm3 = self.cube_size_nm**3
        self.tolerance_nm = float(
            geometric_tolerance_nm
            if geometric_tolerance_nm is not None
            else max(1.0e-9 * self.cube_size_nm, np.finfo(float).eps * 100.0)
        )
        if self.tolerance_nm < 0.0:
            raise ValueError("geometric_tolerance_nm must be non-negative")

        self._edges_by_id: dict[int, EdgeLike] = {}
        for edge in self.edges:
            edge_id = int(edge.id)
            if edge_id in self._edges_by_id:
                raise ValueError(f"duplicate edge id {edge_id}")
            self._validate_segment(edge)
            self._edges_by_id[edge_id] = edge

        sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
        self.points = sampler.random_base2(qmc_power) * self.cube_size_nm
        self.sample_count = len(self.points)
        self.sample_volume_nm3 = self.box_volume_nm3 / self.sample_count
        self._point_tree = cKDTree(
            self.points,
            compact_nodes=True,
            balanced_tree=True,
        )
        self._cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _segment(edge: EdgeLike) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(edge.start_nm, dtype=np.float64)
        end = np.asarray(edge.end_nm, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("edge endpoints must each contain three coordinates")
        return start, end

    @classmethod
    def _validate_segment(cls, edge: EdgeLike) -> None:
        start, end = cls._segment(edge)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError(f"edge {int(edge.id)} has non-finite coordinates")
        if float(np.linalg.norm(end - start)) <= 0.0:
            raise ValueError(f"edge {int(edge.id)} has zero length")

    def _periodic_image_shifts(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> Iterable[tuple[int, int, int]]:
        extension = self.radius_nm + self.tolerance_nm
        lower = np.minimum(start, end) - extension
        upper = np.maximum(start, end) + extension
        ranges: list[range] = []
        for axis in range(3):
            first = int(ceil(-upper[axis] / self.cube_size_nm))
            last = int(floor((self.cube_size_nm - lower[axis]) / self.cube_size_nm))
            ranges.append(range(first, last + 1))
        return product(*ranges)

    def _indices_for_image(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        centre = 0.5 * (start + end)
        query_radius = 0.5 * np.sqrt(length_squared) + self.radius_nm
        candidates = np.asarray(
            self._point_tree.query_ball_point(
                centre,
                query_radius + self.tolerance_nm,
            ),
            dtype=np.int64,
        )
        if not len(candidates):
            return np.empty(0, dtype=np.int32)

        relative = self.points[candidates] - start
        parameter = np.clip((relative @ segment) / length_squared, 0.0, 1.0)
        closest = start + parameter[:, None] * segment
        distance_squared = np.einsum(
            "ij,ij->i",
            self.points[candidates] - closest,
            self.points[candidates] - closest,
        )
        threshold = self.radius_nm + self.tolerance_nm
        return candidates[distance_squared <= threshold * threshold].astype(
            np.int32,
            copy=False,
        )

    def edge_indices(self, edge_id: int) -> np.ndarray:
        """Return and cache sample indices covered by one logical edge."""

        edge_id = int(edge_id)
        cached = self._cache.get(edge_id)
        if cached is not None:
            return cached
        try:
            edge = self._edges_by_id[edge_id]
        except KeyError as exc:
            raise KeyError(f"unknown edge id {edge_id}") from exc

        start, end = self._segment(edge)
        chunks: list[np.ndarray] = []
        for image_shift in self._periodic_image_shifts(start, end):
            translation = np.asarray(image_shift, dtype=np.float64) * self.cube_size_nm
            indices = self._indices_for_image(start + translation, end + translation)
            if len(indices):
                chunks.append(indices)

        if chunks:
            indices = np.unique(np.concatenate(chunks)).astype(np.int32, copy=False)
        else:
            indices = np.empty(0, dtype=np.int32)
        self._cache[edge_id] = indices
        return indices

    def empty_occupancy(self) -> np.ndarray:
        return np.zeros(self.sample_count, dtype=np.bool_)

    def occupancy(self, edge_ids: Iterable[int] = ()) -> np.ndarray:
        occupied = self.empty_occupancy()
        for edge_id in edge_ids:
            self.add_edge(int(edge_id), occupied)
        return occupied

    def marginal(self, edge_id: int, occupancy: np.ndarray) -> int:
        """Number of newly covered samples if ``edge_id`` were added."""

        self._check_occupancy(occupancy)
        indices = self.edge_indices(edge_id)
        return int(np.count_nonzero(~occupancy[indices]))

    def add_edge(self, edge_id: int, occupancy: np.ndarray) -> int:
        """Add an edge in place and return its marginal sample count."""

        self._check_occupancy(occupancy)
        indices = self.edge_indices(edge_id)
        new_indices = indices[~occupancy[indices]]
        occupancy[new_indices] = True
        return int(len(new_indices))

    def union_count(self, edge_ids: Iterable[int]) -> int:
        return int(np.count_nonzero(self.occupancy(edge_ids)))

    def fraction(self, edge_ids: Iterable[int]) -> float:
        return self.fraction_from_count(self.union_count(edge_ids))

    def fraction_from_count(self, count: int) -> float:
        if not isinstance(count, Integral) or not 0 <= int(count) <= self.sample_count:
            raise ValueError("count must lie between zero and sample_count")
        return float(count) / self.sample_count

    def volume_nm3_from_count(self, count: int) -> float:
        return self.fraction_from_count(count) * self.box_volume_nm3

    def nominal_capsule_volume_nm3(self, edge_id: int) -> float:
        edge = self._edges_by_id[int(edge_id)]
        start, end = self._segment(edge)
        length = float(np.linalg.norm(end - start))
        return pi * self.radius_nm**2 * length + 4.0 * pi * self.radius_nm**3 / 3.0

    def _check_occupancy(self, occupancy: np.ndarray) -> None:
        if occupancy.shape != (self.sample_count,) or occupancy.dtype != np.bool_:
            raise ValueError(
                "occupancy must be a boolean array with length sample_count"
            )


@dataclass(frozen=True, slots=True)
class EdgeAttainableRange:
    min_solid_fraction: float
    max_solid_fraction: float
    min_porosity: float
    max_porosity: float
    estimator: str


@dataclass(frozen=True, slots=True)
class EdgeSelectionResult:
    selected_edge_ids: tuple[int, ...]
    backbone_edge_ids: tuple[int, ...]
    status: str
    target_porosity: float
    achieved_porosity: float
    absolute_porosity_error: float
    tolerance: float
    porosity_before_crossing: float | None
    porosity_after_crossing: float | None
    attainable_range: EdgeAttainableRange
    winding: WindingReport
    sample_count: int
    occupied_sample_count: int

    @property
    def within_tolerance(self) -> bool:
        return self.absolute_porosity_error <= self.tolerance

    @property
    def solid_fraction(self) -> float:
        return 1.0 - self.achieved_porosity


def _logical_edge_mapping(
    edges: Sequence[EdgeLike],
    graph: PeriodicGraph,
) -> tuple[dict[int, int], dict[int, set[int]]]:
    logical_ids = {int(edge.id) for edge in edges}
    graph_to_logical: dict[int, int] = {}
    logical_to_graph: dict[int, set[int]] = {edge_id: set() for edge_id in logical_ids}
    for arc in graph.iter_edges():
        candidates = (arc.payload, arc.edge_id)
        logical_id: int | None = None
        for candidate in candidates:
            try:
                converted = int(candidate)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if converted in logical_ids:
                logical_id = converted
                break
        if logical_id is None:
            raise ValueError(
                f"graph edge {arc.edge_id} does not identify a geometry edge"
            )
        graph_to_logical[arc.edge_id] = logical_id
        logical_to_graph[logical_id].add(arc.edge_id)

    missing = [edge_id for edge_id, values in logical_to_graph.items() if not values]
    if missing:
        raise ValueError(f"graph is missing {len(missing)} geometry edges")
    return graph_to_logical, logical_to_graph


def _selected_winding(
    graph: PeriodicGraph,
    selected_logical_ids: set[int],
    logical_to_graph: dict[int, set[int]],
) -> WindingReport:
    graph_edge_ids = {
        graph_edge_id
        for logical_id in selected_logical_ids
        for graph_edge_id in logical_to_graph[logical_id]
    }
    active_nodes: set[Hashable] = set()
    for graph_edge_id in graph_edge_ids:
        arc = graph.edge(graph_edge_id)
        active_nodes.update((arc.source, arc.target))
    return analyze_winding(
        graph,
        active_nodes=active_nodes,
        active_edge_ids=graph_edge_ids,
    )


def select_connected_edges(
    edges: Sequence[EdgeLike],
    graph: PeriodicGraph,
    estimator: CapsuleCoverageEstimator,
    target_porosity: float,
    tolerance: float = 0.005,
    random_seed: int = 0,
) -> EdgeSelectionResult:
    """Select a connected, three-axis-winding random nanorod skeleton.

    A low-volume periodic backbone is protected first.  Complete logical edges
    are then added from its incident frontier using deterministic random
    priorities.  When an addition crosses the target solid fraction, the two
    complete-edge states bracketing the target are compared and the closer one
    is returned.
    """

    if not edges:
        raise ValueError("edges must not be empty")
    if not 0.0 < target_porosity < 1.0:
        raise ValueError("target_porosity must lie strictly between zero and one")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    graph_to_logical, logical_to_graph = _logical_edge_mapping(edges, graph)
    edge_by_id = {int(edge.id): edge for edge in edges}
    full_winding = analyze_winding(graph)
    winding_components = [
        component
        for component in full_winding.components
        if component.full_rank and component.winds_all_axes
    ]
    if not winding_components:
        raise RuntimeError("the Voronoi edge graph has no three-axis winding component")

    component = max(winding_components, key=lambda item: len(item.nodes))
    ordered_roots = sorted(component.nodes, key=repr)
    rng = np.random.default_rng(random_seed)
    if len(ordered_roots) > 8:
        root_indices = np.sort(rng.choice(len(ordered_roots), size=8, replace=False))
        roots = [ordered_roots[int(index)] for index in root_indices]
    else:
        roots = ordered_roots

    def logical_id_for_item(item: Hashable) -> int:
        try:
            candidate = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-integer edge payload {item!r}") from exc
        if candidate in edge_by_id:
            return candidate
        if candidate in graph_to_logical:
            return graph_to_logical[candidate]
        raise ValueError(f"unknown graph edge item {item!r}")

    def nominal_cost(item: Hashable) -> float:
        return max(
            estimator.nominal_capsule_volume_nm3(logical_id_for_item(item)),
            np.finfo(float).tiny,
        )

    backbone = build_axis_backbone(
        graph,
        roots=roots,
        mode="edges",
        allowed_nodes=component.nodes,
        item_cost=nominal_cost,
        try_axis_orders=False,
    )
    if backbone is None:
        raise RuntimeError("could not construct a three-axis periodic edge backbone")

    backbone_graph_ids = set(backbone.edge_ids)
    selected = {graph_to_logical[edge_id] for edge_id in backbone_graph_ids}
    backbone_logical_ids = set(selected)
    active_nodes: set[Hashable] = set()
    for edge_id in backbone_graph_ids:
        arc = graph.edge(edge_id)
        active_nodes.update((arc.source, arc.target))

    # All edges in this quotient-graph component are reachable by connected
    # frontier growth; disconnected components are deliberately excluded.
    component_nodes = set(active_nodes)
    stack = list(active_nodes)
    growable_graph_ids: set[int] = set()
    while stack:
        node = stack.pop()
        for arc in graph.arcs_from(node):
            growable_graph_ids.add(arc.edge_id)
            if arc.target not in component_nodes:
                component_nodes.add(arc.target)
                stack.append(arc.target)
    growable = {graph_to_logical[edge_id] for edge_id in growable_graph_ids}

    occupancy = estimator.occupancy(selected)
    current_count = int(np.count_nonzero(occupancy))
    min_count = current_count
    max_count = estimator.union_count(growable)
    min_solid = estimator.fraction_from_count(min_count)
    max_solid = estimator.fraction_from_count(max_count)
    attainable = EdgeAttainableRange(
        min_solid_fraction=min_solid,
        max_solid_fraction=max_solid,
        min_porosity=1.0 - max_solid,
        max_porosity=1.0 - min_solid,
        estimator=f"scrambled Sobol QMC 2^{int(np.log2(estimator.sample_count))}",
    )

    def make_result(
        chosen: set[int],
        count: int,
        status: str,
        before: float | None = None,
        after: float | None = None,
    ) -> EdgeSelectionResult:
        winding = _selected_winding(graph, chosen, logical_to_graph)
        porosity = 1.0 - estimator.fraction_from_count(count)
        if not winding.connected or not winding.has_three_axis_component:
            status = "TOPOLOGY_FAILURE"
        return EdgeSelectionResult(
            selected_edge_ids=tuple(sorted(chosen)),
            backbone_edge_ids=tuple(sorted(backbone_logical_ids)),
            status=status,
            target_porosity=float(target_porosity),
            achieved_porosity=porosity,
            absolute_porosity_error=abs(porosity - target_porosity),
            tolerance=float(tolerance),
            porosity_before_crossing=before,
            porosity_after_crossing=after,
            attainable_range=attainable,
            winding=winding,
            sample_count=estimator.sample_count,
            occupied_sample_count=int(count),
        )

    if target_porosity > attainable.max_porosity + tolerance:
        return make_result(selected, current_count, "INFEASIBLE_TOO_POROUS")
    if target_porosity < attainable.min_porosity - tolerance:
        return make_result(set(growable), max_count, "INFEASIBLE_TOO_DENSE")

    target_count = int(round((1.0 - target_porosity) * estimator.sample_count))
    if current_count >= target_count:
        porosity = 1.0 - estimator.fraction_from_count(current_count)
        status = (
            "REACHED"
            if abs(porosity - target_porosity) <= tolerance
            else "BRACKETED_DISCRETE"
        )
        return make_result(selected, current_count, status)

    # Assign priorities in sorted order so results do not depend on hash-table
    # iteration or graph construction order.
    priorities = {
        edge_id: float(value)
        for edge_id, value in zip(
            sorted(growable),
            rng.random(len(growable)),
        )
    }
    frontier: set[int] = set()

    def expose(nodes: Iterable[Hashable]) -> None:
        for node in nodes:
            for arc in graph.arcs_from(node):
                logical_id = graph_to_logical[arc.edge_id]
                if logical_id in growable and logical_id not in selected:
                    frontier.add(logical_id)

    expose(active_nodes)
    before_selected: set[int] | None = None
    before_count: int | None = None
    after_selected: set[int] | None = None
    after_count: int | None = None

    while current_count < target_count and frontier:
        chosen = min(frontier, key=lambda edge_id: (priorities[edge_id], edge_id))
        frontier.remove(chosen)
        before_selected = set(selected)
        before_count = current_count
        selected.add(chosen)
        current_count += estimator.add_edge(chosen, occupancy)

        new_nodes: set[Hashable] = set()
        for graph_edge_id in logical_to_graph[chosen]:
            arc = graph.edge(graph_edge_id)
            new_nodes.update((arc.source, arc.target))
        active_nodes.update(new_nodes)
        expose(new_nodes)

        if current_count >= target_count:
            after_selected = set(selected)
            after_count = current_count
            break

    if before_selected is None or before_count is None:
        return make_result(selected, current_count, "FRONTIER_EXHAUSTED")

    before_porosity = 1.0 - estimator.fraction_from_count(before_count)
    if after_selected is None or after_count is None:
        achieved = 1.0 - estimator.fraction_from_count(current_count)
        status = (
            "REACHED"
            if abs(achieved - target_porosity) <= tolerance
            else "FRONTIER_EXHAUSTED"
        )
        return make_result(selected, current_count, status, before_porosity, achieved)

    after_porosity = 1.0 - estimator.fraction_from_count(after_count)
    if abs(before_porosity - target_porosity) <= abs(after_porosity - target_porosity):
        selected, current_count = before_selected, before_count
    else:
        selected, current_count = after_selected, after_count

    achieved = 1.0 - estimator.fraction_from_count(current_count)
    status = (
        "REACHED"
        if abs(achieved - target_porosity) <= tolerance
        else "BRACKETED_DISCRETE"
    )
    return make_result(
        selected,
        current_count,
        status,
        before_porosity,
        after_porosity,
    )


__all__ = [
    "CapsuleCoverageEstimator",
    "EdgeAttainableRange",
    "EdgeSelectionResult",
    "select_connected_edges",
]
