from __future__ import annotations

import re

import numpy as np
import pytest

from aerogel_studio.cad_export import write_faceted_step, write_lumerical_import


def _tetrahedron():
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # Outward faces, signed enclosed volume = 1/6.
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def _cube(lower=0.0, upper=1.0):
    vertices = np.asarray([[lower, lower, lower], [upper, lower, lower],
                           [upper, upper, lower], [lower, upper, lower],
                           [lower, lower, upper], [upper, lower, upper],
                           [upper, upper, upper], [lower, upper, upper]])
    faces = np.asarray([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def _join(*meshes):
    offset, points, triangles = 0, [], []
    for vertices, faces in meshes:
        points.append(vertices)
        triangles.append(faces + offset)
        offset += len(vertices)
    return np.concatenate(points), np.concatenate(triangles)


def _check_references(text):
    entity_ids = re.findall(r"^#(\d+)=", text, re.MULTILINE)
    assert len(entity_ids) == len(set(entity_ids))
    references = set(re.findall(r"#(\d+)", text))
    assert references == set(entity_ids)
    assert text.startswith("ISO-10303-21;\nHEADER;")
    assert text.endswith("ENDSEC;\nEND-ISO-10303-21;\n")


@pytest.mark.parametrize("unit,prefix", [("nm", "NANO"), ("um", "MICRO"), ("mm", "MILLI")])
def test_step_is_closed_solid_with_explicit_units_and_resolved_references(tmp_path, unit, prefix):
    vertices, faces = _tetrahedron()
    path = write_faceted_step(tmp_path / "model.step", vertices, faces, unit=unit)
    text = path.read_text(encoding="ascii")
    _check_references(text)
    assert f"SI_UNIT(.{prefix}.,.METRE.)" in text
    assert len(re.findall(r"=FACE_SURFACE\(", text)) == 4
    assert len(re.findall(r"=PLANE\(", text)) == 4
    assert len(re.findall(r"=POLY_LOOP\(", text)) == 4
    assert len(re.findall(r"=CLOSED_SHELL\(", text)) == 1
    assert len(re.findall(r"=FACETED_BREP\(", text)) == 1
    assert "FACETED_BREP_SHAPE_REPRESENTATION" in text
    assert not path.with_name(path.name + ".partial").exists()


def test_step_keeps_separate_solids_and_corrects_globally_inward_shell(tmp_path):
    vertices, faces = _tetrahedron()
    joined = _join((vertices, faces), (vertices + [3., 0., 0.], faces[:, [0, 2, 1]]))
    text = write_faceted_step(tmp_path / "two.step", *joined).read_text()
    _check_references(text)
    assert len(re.findall(r"=CLOSED_SHELL\(", text)) == 2
    assert len(re.findall(r"=FACETED_BREP\(", text)) == 2
    assert "BREP_WITH_VOIDS" not in text


def test_nested_shell_is_exported_as_cavity_not_filled_solid(tmp_path):
    joined = _join(_cube(0, 3), _cube(1, 2))
    text = write_faceted_step(tmp_path / "hollow.step", *joined).read_text()
    _check_references(text)
    assert len(re.findall(r"=CLOSED_SHELL\(", text)) == 2
    assert len(re.findall(r"=ORIENTED_CLOSED_SHELL\(", text)) == 1
    assert "BREP_WITH_VOIDS" in text
    assert len(re.findall(r"=FACETED_BREP\(", text)) == 0
    assert len(re.findall(r"=BREP_WITH_VOIDS\(", text)) == 1


@pytest.mark.parametrize("damage", ["open", "orientation", "degenerate", "nonfinite"])
def test_step_rejects_non_solid_input_without_creating_output(tmp_path, damage):
    vertices, faces = _tetrahedron()
    if damage == "open":
        faces = faces[:-1]
    elif damage == "orientation":
        faces[0] = faces[0, [0, 2, 1]]
    elif damage == "degenerate":
        vertices[1] = vertices[0]
    else:
        vertices[0, 0] = np.nan
    path = tmp_path / "bad.step"
    with pytest.raises(ValueError):
        write_faceted_step(path, vertices, faces)
    assert not path.exists()
    assert not path.with_name("bad.step.partial").exists()


@pytest.mark.parametrize("unit,scale", [("nm", 1e-9), ("um", 1e-6), ("mm", 1e-3)])
def test_lumerical_import_matches_exported_numeric_unit(tmp_path, unit, scale):
    path = write_lumerical_import(tmp_path / "import.lsf", "aerogel.stl", 1000., unit)
    text = path.read_text(encoding="utf-8")
    value = re.search(r"coordinate_scale_to_m = ([^;]+);", text).group(1)
    assert float(value) == scale
    span = re.search(r"^L = ([^;]+);", text, re.MULTILINE).group(1)
    assert float(span) == pytest.approx(1e-6)
    assert 'stl_file = "aerogel.stl";' in text
    assert "stlimport(stl_file, coordinate_scale_to_m, vertex_merge_radius_m, false)" in text
    assert 'set("x min bc", "Periodic")' in text
    assert 'set("z min bc", "PML")' in text
    assert "placeholder_index" in text


@pytest.mark.parametrize("case,unit,expected,solids", [
    ("tetra", "mm", 1 / 6, 1),
    ("tetra", "um", 1e-9 / 6, 1),
    ("tetra", "nm", 1e-18 / 6, 1),
    ("separate", "mm", 1 / 3, 2),
    ("hollow", "mm", 26., 1),
])
def test_step_roundtrip_with_optional_opencascade(tmp_path, case, unit, expected, solids):
    pytest.importorskip("OCP", exc_type=ImportError)
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID

    vertices, faces = _tetrahedron()
    if case == "separate":
        vertices, faces = _join((vertices, faces), (vertices + [3., 0., 0.], faces))
    elif case == "hollow":
        vertices, faces = _join(_cube(0., 3.), _cube(1., 2.))
    # Keep the physical tetrahedron large enough for the CAD kernel's fixed
    # confusion tolerance while still verifying each declared SI prefix.
    if unit == "nm":
        vertices *= 1000
        expected *= 1000 ** 3
    path = write_faceted_step(tmp_path / "roundtrip.step", vertices, faces, unit)
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    assert reader.TransferRoots() == 1
    shape = reader.OneShape()
    assert not shape.IsNull()
    assert BRepCheck_Analyzer(shape).IsValid()
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    assert properties.Mass() == pytest.approx(expected, rel=1e-8, abs=0.)
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    assert count == solids
