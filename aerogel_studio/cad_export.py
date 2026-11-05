"""Faceted STEP solids and portable Lumerical STL import templates.

STEP files contain the full triangular boundary, not fitted analytic rods.
Closed, consistently oriented edge topology is checked before writing. This
does not replace a target CAD kernel's self-intersection or meshing checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


_UNIT_PREFIX = {"nm": ".NANO.", "um": ".MICRO.", "mm": ".MILLI."}
_UNIT_TO_METRE = {"nm": 1.0e-9, "um": 1.0e-6, "mm": 1.0e-3}


def _real(value: float) -> str:
    """Format a finite STEP REAL, including its required decimal point."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("STEP coordinates must be finite")
    result = format(0.0 if value == 0.0 else value, ".17g").upper()
    if "E" in result:
        mantissa, exponent = result.split("E")
        return mantissa + ("" if "." in mantissa else ".") + "E" + exponent
    return result + ("" if "." in result else ".")


def _mesh_components(vertices: np.ndarray, triangles: np.ndarray):
    vertices = np.asarray(vertices, dtype=float)
    indices = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a nonempty (N, 3) array")
    if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices):
        raise ValueError("triangles must be a nonempty (M, 3) array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("triangle indices must be integers")
    triangles = indices.astype(np.int64, copy=False)
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or infinite values")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle indices are outside the vertex array")
    used, inverse_vertices = np.unique(triangles, return_inverse=True)
    vertices = vertices[used]
    triangles = inverse_vertices.reshape((-1, 3))
    tri = vertices[triangles]
    if np.any(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) == 0):
        raise ValueError("STEP cannot contain degenerate triangles")
    del tri

    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    _, inverse_edges, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("STEP requires a watertight manifold mesh: every edge must have exactly two faces")
    orientation = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(np.bincount(inverse_edges, weights=orientation) != 0):
        raise ValueError("STEP requires consistently oriented neighbouring triangles")
    edge_order = np.argsort(inverse_edges, kind="stable").reshape((-1, 2))
    face_ids = np.tile(np.arange(len(triangles)), 3)
    pairs = face_ids[edge_order]
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(triangles), len(triangles))).tocsr()
    count, labels = connected_components(graph, directed=False)
    del directed, inverse_edges, counts, orientation, edge_order, face_ids, pairs, graph

    groups = np.split(np.argsort(labels, kind="stable"), np.flatnonzero(np.diff(np.sort(labels))) + 1)
    components = []
    for group in groups:
        points = vertices[triangles[group]]
        origin = points[0, 0]
        shifted = points - origin
        volume = float(np.einsum("ij,ij->i", shifted[:, 0],
                                np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6.0)
        if volume == 0.0:
            raise ValueError("a closed STEP shell encloses zero signed volume")
        components.append({"faces": group, "volume": volume,
                           "minimum": points.min(axis=(0, 1)),
                           "maximum": points.max(axis=(0, 1)),
                           "point": points[0, 0]})
    assert len(components) == count
    return vertices, triangles, labels, components


def _inside(point: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> bool:
    """Ray parity with three generic directions; merge shared-edge hits."""
    span = float(np.ptp(vertices, axis=0).max())
    tolerance = max(span * 1.0e-10, np.finfo(float).tiny)
    votes = 0
    for direction in ((1.0, 0.371390676, 0.693147181),
                      (0.51763809, 1.0, 0.271828182),
                      (0.414213562, 0.618033989, 1.0)):
        ray = np.asarray(direction, dtype=float)
        ray /= np.linalg.norm(ray)
        hits = []
        for start in range(0, len(triangles), 50_000):
            tri = vertices[triangles[start:start + 50_000]]
            edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            pvec = np.cross(np.broadcast_to(ray, edge2.shape), edge2)
            determinant = np.einsum("ij,ij->i", edge1, pvec)
            valid = np.abs(determinant) > np.finfo(float).eps * np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
            inv = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
            tvec = point - tri[:, 0]
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            qvec = np.cross(tvec, edge1)
            v = (qvec @ ray) * inv
            distance = np.einsum("ij,ij->i", edge2, qvec) * inv
            hit = valid & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance > tolerance)
            hits.extend(distance[hit].tolist())
        if hits:
            distances = np.sort(hits)
            distinct_count = 1 + int(np.count_nonzero(np.diff(distances) > tolerance))
            votes += distinct_count % 2
    return votes >= 2


def _parents(vertices, triangles, components):
    """Classify nested shells so cavities remain voids rather than solids."""
    parents = [-1] * len(components)
    for index, child in enumerate(components):
        candidates = [other for other, candidate in enumerate(components)
                      if other != index and abs(candidate["volume"]) > abs(child["volume"])
                      and np.all(child["minimum"] >= candidate["minimum"])
                      and np.all(child["maximum"] <= candidate["maximum"])]
        for other in sorted(candidates, key=lambda i: abs(components[i]["volume"])):
            if _inside(child["point"], vertices, triangles[components[other]["faces"]]):
                parents[index] = other
                break
    depths = []
    for index in range(len(components)):
        depth, ancestor = 0, parents[index]
        while ancestor != -1:
            depth += 1
            ancestor = parents[ancestor]
        depths.append(depth)
    return parents, depths


def _references(stream: TextIO, values) -> None:
    for start in range(0, len(values), 16):
        if start:
            stream.write(",\n")
        stream.write(",".join(f"#{int(value)}" for value in values[start:start + 16]))


def write_faceted_step(path: str | Path, vertices, triangles, unit: str = "um") -> Path:
    """Write AP203 planar-faceted solids, preserving solids and interior voids.

    ``vertices`` must already be scaled into ``unit``. A triangle becomes one
    planar CAD face, so large scientific meshes also produce large STEP files.
    No analytic cylinder fitting or CAD-kernel healing is performed.
    """
    if unit not in _UNIT_PREFIX:
        raise ValueError("STEP unit must be nm, um or mm")
    vertices, triangles, labels, components = _mesh_components(vertices, triangles)
    parents, depths = _parents(vertices, triangles, components)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filename = target.name.encode("ascii", "replace").decode("ascii").replace("'", "''")
    uncertainty = max(float(np.ptp(vertices, axis=0).max()) * 1e-10, 1e-15)
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("ISO-10303-21;\nHEADER;\n")
            stream.write(f"FILE_DESCRIPTION(('Aerogel Studio faceted solid; unit={unit}'),'2;1');\n")
            stream.write(f"FILE_NAME('{filename}','{timestamp}',('Aerogel Studio'),(''),'Aerogel Studio','Python faceted STEP exporter','');\n")
            stream.write("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\nENDSEC;\nDATA;\n")
            stream.write("#1=APPLICATION_CONTEXT('configuration controlled 3d designs of mechanical parts and assemblies');\n")
            stream.write("#2=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',1994,#1);\n")
            stream.write("#3=PRODUCT_CONTEXT('',#1,'mechanical');\n")
            stream.write("#4=PRODUCT('Aerogel','Aerogel Studio solid','',(#3));\n")
            stream.write("#5=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#4,.NOT_KNOWN.);\n")
            stream.write("#6=PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');\n")
            stream.write("#7=PRODUCT_DEFINITION('design','',#5,#6);\n")
            stream.write("#8=PRODUCT_DEFINITION_SHAPE('','',#7);\n")
            stream.write(f"#9=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({_UNIT_PREFIX[unit]},.METRE.));\n")
            stream.write("#10=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));\n")
            stream.write("#11=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());\n")
            stream.write(f"#12=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_real(uncertainty)}),#9,'distance_accuracy_value','mesh precision');\n")
            stream.write("#13=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#12)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#9,#10,#11)) REPRESENTATION_CONTEXT('',''));\n")
            first_point = 14
            for point_id, point in enumerate(vertices, start=first_point):
                stream.write(f"#{point_id}=CARTESIAN_POINT('',({','.join(_real(x) for x in point)}));\n")
            first_loop = first_point + len(vertices)
            outward = np.asarray([component["volume"] >= 0 for component in components])
            for index, triangle in enumerate(triangles):
                if not outward[labels[index]]:
                    triangle = triangle[[0, 2, 1]]
                loop = first_loop + 7 * index
                points = ",".join(f"#{first_point + int(vertex)}" for vertex in triangle)
                tri_points = vertices[triangle]
                edge = tri_points[1] - tri_points[0]
                reference = edge / np.linalg.norm(edge)
                normal = np.cross(edge, tri_points[2] - tri_points[0])
                normal /= np.linalg.norm(normal)
                stream.write(f"#{loop}=POLY_LOOP('',({points}));\n")
                stream.write(f"#{loop + 1}=FACE_OUTER_BOUND('',#{loop},.T.);\n")
                stream.write(f"#{loop + 2}=DIRECTION('',({','.join(_real(x) for x in normal)}));\n")
                stream.write(f"#{loop + 3}=DIRECTION('',({','.join(_real(x) for x in reference)}));\n")
                stream.write(f"#{loop + 4}=AXIS2_PLACEMENT_3D('',#{first_point + int(triangle[0])},#{loop + 2},#{loop + 3});\n")
                stream.write(f"#{loop + 5}=PLANE('',#{loop + 4});\n")
                stream.write(f"#{loop + 6}=FACE_SURFACE('',(#{loop + 1}),#{loop + 5},.T.);\n")
            next_id = first_loop + 7 * len(triangles)
            shell_ids = []
            for component in components:
                shell_ids.append(next_id)
                stream.write(f"#{next_id}=CLOSED_SHELL('',(\n")
                _references(stream, first_loop + 7 * component["faces"] + 6)
                stream.write("));\n")
                next_id += 1
            solid_ids = []
            for index, shell in enumerate(shell_ids):
                if depths[index] % 2:
                    continue
                voids = [child for child in range(len(components)) if parents[child] == index]
                void_ids = []
                for child in voids:
                    stream.write(f"#{next_id}=ORIENTED_CLOSED_SHELL('',*,#{shell_ids[child]},.F.);\n")
                    void_ids.append(next_id)
                    next_id += 1
                solid_ids.append(next_id)
                if void_ids:
                    refs = ",".join(f"#{value}" for value in void_ids)
                    # A direct BREP_WITH_VOIDS in a generic representation is
                    # used here. OCCT 8's faceted-BREP complex importer drops
                    # its voids, whereas this form retains all cavity shells.
                    stream.write(f"#{next_id}=BREP_WITH_VOIDS('Aerogel solid',#{shell},({refs}));\n")
                else:
                    stream.write(f"#{next_id}=FACETED_BREP('Aerogel solid',#{shell});\n")
                next_id += 1
            refs = ",".join(f"#{value}" for value in solid_ids)
            representation = "SHAPE_REPRESENTATION" if any(depth % 2 for depth in depths) else "FACETED_BREP_SHAPE_REPRESENTATION"
            stream.write(f"#{next_id}={representation}('',({refs}),#13);\n")
            stream.write(f"#{next_id + 1}=SHAPE_DEFINITION_REPRESENTATION(#8,#{next_id});\n")
            stream.write("ENDSEC;\nEND-ISO-10303-21;\n")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_lumerical_import(path: str | Path, stl_filename: str | Path,
                          cube_size_nm: float, export_unit: str) -> Path:
    """Write a unit-correct STL import and a finite-slab FDTD setup template.

    Syntax and scale follow the Ansys stlimport documentation:
    https://optics.ansys.com/hc/en-us/articles/360034924733-stlimport-Script-command
    """
    if export_unit not in _UNIT_TO_METRE:
        raise ValueError("export_unit must be nm, um or mm")
    if not np.isfinite(cube_size_nm) or cube_size_nm <= 0:
        raise ValueError("cube_size_nm must be finite and positive")
    filename = Path(stl_filename).as_posix().replace('"', '\\"')
    scale = _UNIT_TO_METRE[export_unit]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    script = f'''# Aerogel Studio - Lumerical FDTD import template.
# STL numeric coordinate unit: {export_unit}; scale below converts to metres.
# Run from this export directory in a fresh layout. This script imports
# geometry and a simulation region, but does not start a simulation.
# Select a validated dispersive material, source, monitors and mesh for your
# wavelength band before quantitative use. The placeholder index is not a
# fitted alumina optical material model.
switchtolayout;
stl_file = "{filename}";
coordinate_scale_to_m = {scale:.12g};
vertex_merge_radius_m = 1e-12;
# If invalid triangles are reported, inspect geometry resolution and the
# vertex merge radius. Never choose a radius comparable to a thin strut.
stlimport(stl_file, coordinate_scale_to_m, vertex_merge_radius_m, false);
set("name", "aerogel_solid");
material_name = "<Object defined dielectric>"; # Edit for your material database.
placeholder_index = 1.76;
set("material", material_name);
if (material_name == "<Object defined dielectric>") {{
    set("index", placeholder_index);
}}

L = {cube_size_nm * 1e-9:.17g};
addfdtd;
set("name", "FDTD_aerogel_cell");
set("dimension", "3D");
set("x", 0.5*L);
set("y", 0.5*L);
set("z", 0.5*L);
set("x span", L);
set("y span", L);
set("z span", 1.5*L);
set("x min bc", "Periodic");
set("x max bc", "Periodic");
set("y min bc", "Periodic");
set("y max bc", "Periodic");
set("z min bc", "PML");
set("z max bc", "PML");
# x/y Periodic assumes normal incidence. Use Bloch conditions for oblique
# incidence. z PML represents a finite slab; geometry has a vacuum buffer.
# Add source wavelength range and monitors; check mesh convergence against
# the minimum rod diameter / sheet thickness and CAD voxel resolution.
'''
    target.write_text(script, encoding="utf-8", newline="\n")
    return target


__all__ = ["write_faceted_step", "write_lumerical_import"]
