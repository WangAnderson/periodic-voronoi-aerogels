"""Periodic Voronoi edges used as nanowire centre lines.

The sheet model stores complete, unwrapped Voronoi faces.  This module derives
one canonical logical edge from every set of periodically equivalent polygon
edges.  Topology is kept before clipping: a boundary-crossing edge is one
graph edge with an integer lattice ``shift``, not two unrelated line pieces.

``start_nm`` is the wrapped representative of ``start_vertex`` in the primary
box.  ``end_nm`` is the matching periodic image of ``end_vertex`` and may lie
outside the box.  Their Euclidean distance is therefore the physical length of
the complete logical edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import NDArray

from voronoi_aerogel.topology import PeriodicGraph


FloatArray: TypeAlias = NDArray[np.float64]
Int3: TypeAlias = tuple[int, int, int]
EdgeKey: TypeAlias = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class PeriodicEdge:
    """One canonical Voronoi edge in the periodic covering space.

    ``shift`` is the lattice displacement encountered when traversing from
    ``start_vertex`` to ``end_vertex``.  ``incident_face_ids`` records every
    input Voronoi face sharing the edge.
    """

    id: int
    start_vertex: int
    end_vertex: int
    shift: Int3
    start_nm: FloatArray
    end_nm: FloatArray
    length_nm: float
    incident_face_ids: tuple[int, ...]


class _PeriodicVertexRegistry:
    """Map lifted coordinates to a primary-box vertex ID and lattice lift."""

    def __init__(self, cube_size_nm: float, tolerance_nm: float) -> None:
        self.cube_size_nm = cube_size_nm
        self.tolerance_nm = tolerance_nm
        self.positions: list[FloatArray] = []
        self._buckets: dict[Int3, list[int]] = {}

    def register(self, point: FloatArray) -> tuple[int, NDArray[np.int64]]:
        wrapped = np.mod(point, self.cube_size_nm)
        wrapped[np.abs(wrapped) <= self.tolerance_nm] = 0.0
        wrapped[
            np.abs(wrapped - self.cube_size_nm) <= self.tolerance_nm
        ] = 0.0
        bucket = tuple(
            int(value)
            for value in np.rint(wrapped / self.tolerance_nm).astype(np.int64)
        )

        vertex_id: int | None = None
        for offset in product((-1, 0, 1), repeat=3):
            neighbour = tuple(bucket[axis] + offset[axis] for axis in range(3))
            for candidate_id in self._buckets.get(neighbour, ()):
                delta = np.abs(wrapped - self.positions[candidate_id])
                delta = np.minimum(delta, self.cube_size_nm - delta)
                if float(np.linalg.norm(delta)) <= self.tolerance_nm:
                    vertex_id = candidate_id
                    wrapped = self.positions[candidate_id]
                    break
            if vertex_id is not None:
                break

        if vertex_id is None:
            vertex_id = len(self.positions)
            self.positions.append(wrapped.copy())
            self._buckets.setdefault(bucket, []).append(vertex_id)

        lift = np.rint((point - wrapped) / self.cube_size_nm).astype(np.int64)
        reconstructed = wrapped + lift * self.cube_size_nm
        if float(np.linalg.norm(point - reconstructed)) > 2.0 * self.tolerance_nm:
            raise RuntimeError("Could not assign a consistent periodic vertex lift")
        return vertex_id, lift


def extract_periodic_edges(
    faces: Iterable[object],
    cube_size_nm: float,
    tolerance_nm: float | None = None,
) -> list[PeriodicEdge]:
    """Extract unique logical edges from ordered periodic Voronoi faces.

    Each input face must provide ``id`` and an ordered ``vertices_nm`` array.
    Adjacent polygon vertices are registered as ``(periodic vertex ID, lift)``.
    The undirected key consists of the two canonical vertex IDs and the
    relative lift, so occurrences on different periodic copies collapse while
    a true boundary-crossing edge retains its winding shift.
    """

    if cube_size_nm <= 0.0:
        raise ValueError("cube_size_nm must be positive")
    tolerance = _resolve_tolerance(cube_size_nm, tolerance_nm)
    registry = _PeriodicVertexRegistry(float(cube_size_nm), tolerance)
    incidents: dict[EdgeKey, set[int]] = {}

    for face in faces:
        vertices = np.asarray(getattr(face, "vertices_nm"), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("each face.vertices_nm must have shape (M, 3), M >= 3")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("face vertices must contain only finite values")
        face_id = int(getattr(face, "id"))
        references = [registry.register(vertex) for vertex in vertices]

        for index in range(len(vertices)):
            vertex_a, lift_a = references[index]
            vertex_b, lift_b = references[(index + 1) % len(vertices)]
            relative_lift = lift_b - lift_a
            key = _canonical_edge_key(vertex_a, vertex_b, relative_lift)
            incidents.setdefault(key, set()).add(face_id)

    edges: list[PeriodicEdge] = []
    for edge_id, key in enumerate(sorted(incidents)):
        start_vertex, end_vertex, shift_x, shift_y, shift_z = key
        shift = (shift_x, shift_y, shift_z)
        start = registry.positions[start_vertex].copy()
        end = (
            registry.positions[end_vertex]
            + np.asarray(shift, dtype=np.float64) * cube_size_nm
        )
        length = float(np.linalg.norm(end - start))
        if length <= tolerance:
            raise RuntimeError(f"Degenerate periodic Voronoi edge for key {key}")
        edges.append(
            PeriodicEdge(
                id=edge_id,
                start_vertex=start_vertex,
                end_vertex=end_vertex,
                shift=shift,
                start_nm=start,
                end_nm=end,
                length_nm=length,
                incident_face_ids=tuple(sorted(incidents[key])),
            )
        )
    return edges


def build_edge_graph(
    edges: Iterable[PeriodicEdge],
    vertex_count: int | None = None,
) -> PeriodicGraph:
    """Build the quotient graph of periodic Voronoi vertices and edges.

    Edge IDs are preserved both as graph edge IDs and payloads.  Supplying
    ``vertex_count`` also inserts isolated vertices ``0 .. vertex_count - 1``;
    otherwise the graph contains exactly the vertices referenced by ``edges``.
    """

    edge_list = list(edges)
    if vertex_count is not None:
        if vertex_count < 0:
            raise ValueError("vertex_count must be non-negative")
        if any(
            edge.start_vertex >= vertex_count or edge.end_vertex >= vertex_count
            for edge in edge_list
        ):
            raise ValueError("vertex_count does not include every edge endpoint")
        graph = PeriodicGraph(range(vertex_count))
    else:
        graph = PeriodicGraph()

    for edge in edge_list:
        graph.add_edge(
            edge.start_vertex,
            edge.end_vertex,
            edge.shift,
            payload=edge.id,
            edge_id=edge.id,
        )
    return graph


def periodic_edge_segments(
    edge: PeriodicEdge,
    cube_size_nm: float,
    tolerance_nm: float | None = None,
) -> list[FloatArray]:
    """Translate and clip a logical edge into the primary cube for display.

    Each returned ``(2, 3)`` array is one visible centre-line segment inside
    ``[0, cube_size_nm]^3``.  Selection and topology must use the original
    :class:`PeriodicEdge`; these pieces are only periodic display/export views.
    """

    if cube_size_nm <= 0.0:
        raise ValueError("cube_size_nm must be positive")
    tolerance = _resolve_tolerance(cube_size_nm, tolerance_nm)
    start = np.asarray(edge.start_nm, dtype=np.float64)
    end = np.asarray(edge.end_nm, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("edge endpoints must each have shape (3,)")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("edge endpoints must contain only finite values")

    minimum = np.minimum(start, end)
    maximum = np.maximum(start, end)
    shift_ranges: list[range] = []
    for axis in range(3):
        first = int(ceil((-maximum[axis] - tolerance) / cube_size_nm))
        last = int(
            floor((cube_size_nm - minimum[axis] + tolerance) / cube_size_nm)
        )
        shift_ranges.append(range(first, last + 1))

    segments: list[FloatArray] = []
    seen: set[tuple[int, ...]] = set()
    for tile_shift in product(*shift_ranges):
        translation = np.asarray(tile_shift, dtype=np.float64) * cube_size_nm
        clipped = _clip_segment_to_box(
            start + translation,
            end + translation,
            cube_size_nm,
            tolerance,
        )
        if clipped is None:
            continue
        clipped[np.abs(clipped) <= tolerance] = 0.0
        clipped[np.abs(clipped - cube_size_nm) <= tolerance] = cube_size_nm
        if float(np.linalg.norm(clipped[1] - clipped[0])) <= tolerance:
            continue
        key = _segment_key(clipped, tolerance)
        if key not in seen:
            seen.add(key)
            segments.append(clipped)
    return segments


def _canonical_edge_key(
    vertex_a: int,
    vertex_b: int,
    relative_lift: NDArray[np.int64],
) -> EdgeKey:
    if vertex_a < vertex_b:
        return (vertex_a, vertex_b, *map(int, relative_lift))
    if vertex_b < vertex_a:
        return (vertex_b, vertex_a, *map(int, -relative_lift))

    winding = tuple(int(value) for value in relative_lift)
    canonical_winding = min(winding, tuple(-value for value in winding))
    return (vertex_a, vertex_a, *canonical_winding)


def _clip_segment_to_box(
    start: FloatArray,
    end: FloatArray,
    cube_size_nm: float,
    tolerance: float,
) -> FloatArray | None:
    direction = end - start
    lower_t = 0.0
    upper_t = 1.0
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) <= tolerance:
            if start[axis] < -tolerance or start[axis] > cube_size_nm + tolerance:
                return None
            continue
        enter = (0.0 - start[axis]) / component
        leave = (cube_size_nm - start[axis]) / component
        if enter > leave:
            enter, leave = leave, enter
        lower_t = max(lower_t, float(enter))
        upper_t = min(upper_t, float(leave))
        if lower_t > upper_t + tolerance / cube_size_nm:
            return None
    return np.asarray(
        [start + lower_t * direction, start + upper_t * direction],
        dtype=np.float64,
    )


def _segment_key(segment: FloatArray, tolerance: float) -> tuple[int, ...]:
    quantized = np.rint(segment / tolerance).astype(np.int64)
    forward = tuple(int(value) for value in quantized.ravel())
    reverse = tuple(int(value) for value in quantized[::-1].ravel())
    return min(forward, reverse)


def _resolve_tolerance(cube_size_nm: float, tolerance_nm: float | None) -> float:
    if tolerance_nm is None:
        return max(
            1.0e-9 * cube_size_nm,
            np.finfo(np.float64).eps * cube_size_nm * 100,
        )
    if tolerance_nm <= 0.0:
        raise ValueError("tolerance_nm must be positive")
    return float(tolerance_nm)


__all__ = [
    "PeriodicEdge",
    "build_edge_graph",
    "extract_periodic_edges",
    "periodic_edge_segments",
]
