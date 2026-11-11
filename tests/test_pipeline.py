"""Proposers, repair loop and disassembly on the synthetic samples."""
import json

import pytest

pytest.importorskip("OCC")

from exploded_view_verifier import geometry as G  # noqa: E402
from exploded_view_verifier import oracle as O  # noqa: E402
from exploded_view_verifier import samples as S  # noqa: E402
from exploded_view_verifier.cli import main  # noqa: E402
from exploded_view_verifier.disassembly import sequential_disassembly  # noqa: E402
from exploded_view_verifier.proposers import (RandomProposer, ResidualContext,  # noqa: E402
                                              centroid_plan, connected_components)
from exploded_view_verifier.repair import explode, residual_loop  # noqa: E402


def test_assembled_samples():
    for name in ("pin_in_plate", "shaft_stack"):
        assert O.verify_plan(S.build(name), {}), name       # parts touch when assembled
    assert O.verify_plan(S.build("caged_cube"), {}) == []    # the cube floats in the cavity


def test_pin_in_plate_is_repaired_and_disassemblable():
    asm = S.build("pin_in_plate")
    res = explode(asm, check_disassembly=True)
    assert res.accepted
    assert res.disassembly_feasible
    assert sorted(res.removal_order) == ["pin", "plate"]


def test_free_final_state_but_no_disassembly():
    asm = S.build("caged_cube")
    res = explode(asm, check_disassembly=True)
    assert res.accepted                       # the exploded state has no interference
    assert res.disassembly_feasible is False  # but the cube cannot leave a closed box
    assert set(res.stuck_parts) == {"inner", "shell"}


def test_shaft_stack_pipeline_reports_consistent_result():
    asm = S.build("shaft_stack")
    res = explode(asm, proposer=RandomProposer(seed=1))
    final = O.verify_plan(asm, res.plan)
    assert sorted(final) == sorted(res.conflicts)   # the reported status is the verifier's
    assert res.history[0]["step"] == "centroid"


def test_random_proposer_is_reproducible_and_only_moves_conflicting_parts():
    asm = S.build("shaft_stack")
    plan = centroid_plan(asm)
    ctx = ResidualContext(assembly=asm, plan=plan, start=plan,
                          conflicts=[("ring_1", "ring_2")], round=1)
    a, b = RandomProposer(3)(ctx), RandomProposer(3)(ctx)
    assert a == b
    assert set(a) == {"ring_1", "ring_2"}


def test_residual_loop_ignores_vectors_for_other_parts():
    asm = S.build("pin_in_plate")
    calls = []

    def proposer(ctx):
        calls.append(ctx.round)
        return {"not_a_part": [1, 2, 3]}

    plan, coll, n = residual_loop(asm, {}, [("plate", "pin")], proposer, budget=2)
    assert calls == [1, 2] and n == 0 and coll == [("plate", "pin")]


def test_connected_components():
    comps = connected_components([("a", "b"), ("b", "c"), ("x", "y")])
    assert sorted(map(tuple, comps)) == [("a", "b", "c"), ("x", "y")]


def test_sequential_disassembly_blocked_without_vector():
    asm = S.build("pin_in_plate")
    ok, order, stuck = sequential_disassembly(asm, {})
    assert not ok and order == [] and stuck == ["pin", "plate"]


def test_cli_roundtrip(tmp_path):
    main(["--write-samples", str(tmp_path)])
    step = tmp_path / "pin_in_plate.step"
    assert step.exists()
    assert len(G.load_step_assembly(str(step))) == 2
    out = tmp_path / "plan.json"
    assert main([str(step), "--proposer", "none", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["accepted"] is True
    assert main([str(step), "--check", str(out)]) == 0
