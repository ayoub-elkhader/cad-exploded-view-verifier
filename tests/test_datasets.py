"""Dataset loaders on small synthetic files in the Fusion / AutoMate formats, and the
bundled real examples (AutoMate CC0, Zenodo CC BY 4.0)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("OCC")

from exploded_view_verifier import geometry as G  # noqa: E402
from exploded_view_verifier.datasets import (automate_part_ids, load_automate,  # noqa: E402
                                             load_fusion)
from exploded_view_verifier.repair import explode  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def box(dx, dy, dz):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def lo(shapes):
    return [round(v, 3) for v in G.combined_bbox(shapes).Get()[:3]]


def frame(x=0.0, y=0.0, z=0.0):
    return {"origin": {"x": x, "y": y, "z": z}, "x_axis": {"x": 1, "y": 0, "z": 0},
            "y_axis": {"x": 0, "y": 1, "z": 0}, "z_axis": {"x": 0, "y": 0, "z": 1}}


def test_fusion_transforms_are_composed_along_the_tree(tmp_path):
    G.write_step([box(10, 10, 10)], tmp_path / "b1.step")
    G.write_step([box(5, 5, 5)], tmp_path / "b2.step")
    data = {
        "bodies": {"b1": {"step": "b1.step"}, "b2": {"step": "b2.step"}},
        "occurrences": {
            "parent": {"name": "Parent", "transform": frame(x=3.0), "bodies": {"b1": {}}},
            "child": {"name": "Child", "transform": frame(z=2.0), "bodies": {"b2": {}}},
            "missing": {"name": "No geometry", "bodies": {"b9": {}}},
        },
        "tree": {"root": {"parent": {"child": {}}}},
    }
    (tmp_path / "assembly.json").write_text(json.dumps(data))
    asm, names = load_fusion(str(tmp_path / "assembly.json"))
    assert set(asm) == {"parent", "child"}
    assert names == {"parent": "Parent", "child": "Child"}
    assert lo(asm["parent"]) == pytest.approx([30, 0, 0], abs=0.01)   # 3 cm -> 30 mm
    assert lo(asm["child"]) == pytest.approx([30, 0, 20], abs=0.01)   # parent then local


def test_fusion_bodies_only_fallback(tmp_path):
    G.write_step([box(10, 10, 10)], tmp_path / "b1.step")
    G.write_step([box(5, 5, 5)], tmp_path / "b2.step")
    data = {"bodies": {"b1": {"name": "A"}, "b2": {"name": "B"}}, "occurrences": {}}
    (tmp_path / "assembly.json").write_text(json.dumps(data))
    asm, names = load_fusion(str(tmp_path / "assembly.json"))
    assert names == {"b1": "A", "b2": "B"}


def test_automate_transforms_and_filters(tmp_path):
    G.write_step([box(10, 10, 10)], tmp_path / "pa.step")
    G.write_step([box(5, 5, 5)], tmp_path / "pb.step")
    eye = lambda tx: [1, 0, 0, tx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # noqa: E731
    asm_json = {
        "parts": [{"id": "pa", "has_step": True, "name": "plate"}, {"id": "pb", "has_step": True},
                  {"id": "pc", "has_step": False}],
        "occurrences": [
            {"id": "o1", "part": 0, "transform": eye(0.0)},
            {"id": "o2", "part": 1, "transform": eye(0.02)},          # 0.02 m -> 20 mm
            {"id": "o3", "part": 1, "transform": eye(0.05), "hidden": True},
            {"id": "o4", "part": 2, "transform": eye(0.0)},
        ],
    }
    p = tmp_path / "asm.json"
    p.write_text(json.dumps(asm_json))
    asm, names = load_automate(str(p), str(tmp_path))
    assert set(asm) == {"o1", "o2"}
    assert names["o1"] == "plate"
    assert lo(asm["o2"]) == pytest.approx([20, 0, 0], abs=0.01)
    assert automate_part_ids(str(p)) == ["pa", "pb"]


@pytest.mark.parametrize("prefix,parts,first_conflicts", [
    ("0c4c5dd5", 3, 0), ("0b7302d4", 4, 1), ("d0582033", 8, 3)])
def test_bundled_automate_examples(prefix, parts, first_conflicts):
    path = next((EXAMPLES / "automate").glob(prefix + "*.json"))
    asm, _ = load_automate(str(path), str(EXAMPLES / "automate" / "step"))
    assert len(asm) == parts
    res = explode(asm, check_disassembly=True)
    assert res.history[0]["conflicts"] == first_conflicts
    assert res.accepted and res.disassembly_feasible


def test_bundled_zenodo_example():
    asm = G.load_step_assembly(str(EXAMPLES / "zenodo" / "aa-swn2.step"))
    assert len(asm) == 3
    assert explode(asm).accepted
