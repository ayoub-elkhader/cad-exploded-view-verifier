"""Command-line interface.

    exploded-view-verify assembly.step [--proposer random|scale|llm|none]
                         [--disassembly] [--repair-disassembly] [--out plan.json]
    exploded-view-verify --sample shaft_stack
    exploded-view-verify --write-samples examples/
    exploded-view-verify assembly.step --check plan.json
    exploded-view-verify --fusion a1.0.0_04/53060_3cef188a/assembly.json
    exploded-view-verify --automate automate_asm/<id>.json --step-dir automate_step/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import geometry as G
from . import oracle as O
from . import samples as S
from .proposers import RandomProposer, ScaleProposer
from .repair import explode


def _proposer(name, seed):
    if name == "none":
        return None
    if name == "random":
        return RandomProposer(seed)
    if name == "scale":
        return ScaleProposer()
    if name == "llm":
        from .llm import LLMProposer
        return LLMProposer()
    raise SystemExit(f"unknown proposer {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exploded-view-verify",
                                 description="Explode a STEP assembly and verify the result.")
    ap.add_argument("step", nargs="?", help="STEP file (one solid = one part)")
    ap.add_argument("--sample", choices=sorted(S.SAMPLES), help="use a built-in synthetic assembly")
    ap.add_argument("--fusion", metavar="ASSEMBLY_JSON", help="Fusion 360 Gallery assembly.json")
    ap.add_argument("--automate", metavar="ASSEMBLY_JSON", help="AutoMate assembly JSON (needs --step-dir)")
    ap.add_argument("--step-dir", help="folder of AutoMate part STEP files")
    ap.add_argument("--write-samples", metavar="DIR", help="write the built-in samples as STEP files and exit")
    ap.add_argument("--check", metavar="PLAN_JSON", help="only verify an existing plan")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=G.TOL)
    ap.add_argument("--deflection", type=float, default=G.DEFLECTION)
    ap.add_argument("--proposer", default="random", choices=["none", "random", "scale", "llm"],
                    help="residual proposer used after the deterministic spreading (default: random)")
    ap.add_argument("--budget", type=int, default=4, help="number of residual proposals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disassembly", action="store_true", help="test sequential disassembly")
    ap.add_argument("--repair-disassembly", action="store_true", help="also repair it")
    ap.add_argument("--out", help="write the result as JSON")
    a = ap.parse_args(argv)

    if a.write_samples:
        os.makedirs(a.write_samples, exist_ok=True)
        for name, fn in S.SAMPLES.items():
            path = os.path.join(a.write_samples, f"{name}.step")
            G.write_step([s for sh in fn().values() for s in sh], path)
            print(path)
        return 0

    if a.sample:
        assembly, case = S.build(a.sample, a.deflection), a.sample
    elif a.fusion:
        from .datasets import load_fusion
        assembly, _ = load_fusion(a.fusion, deflection=a.deflection)
        case = os.path.basename(os.path.dirname(os.path.abspath(a.fusion)))
    elif a.automate:
        if not a.step_dir:
            ap.error("--automate needs --step-dir")
        from .datasets import load_automate
        assembly, _ = load_automate(a.automate, a.step_dir, deflection=a.deflection)
        case = os.path.splitext(os.path.basename(a.automate))[0]
    elif a.step:
        assembly, case = G.load_step_assembly(a.step, a.deflection), os.path.basename(a.step)
    else:
        ap.error("give a STEP file, --sample, --fusion or --automate")
    if len(assembly) < 2:
        print(f"{case}: fewer than 2 parts loaded, nothing to explode")
        return 2
    print(f"{case}: {len(assembly)} parts, oracle {O.ORACLE_VERSION}, tol {a.tol} mm")

    if a.check:
        with open(a.check, encoding="utf-8") as f:
            plan = json.load(f)
        plan = plan.get("plan", plan)
        details = {}
        coll = O.verify_plan(assembly, plan, a.tol, a.deflection, details=details)
        for (p, q), v in details.items():
            print(f"  {p} <-> {q}: {v}")
        print("ACCEPTED" if not coll else f"REJECTED ({len(coll)} non-free pairs)")
        return 0 if not coll else 1

    res = explode(assembly, a.factor, a.tol, a.deflection,
                  proposer=_proposer(a.proposer, a.seed), budget=a.budget,
                  check_disassembly=a.disassembly, repair_disassembly=a.repair_disassembly,
                  case_id=case)
    for h in res.history:
        print("  " + ", ".join(f"{k}={v}" for k, v in h.items()))
    print(("ACCEPTED" if res.accepted else f"REJECTED ({len(res.conflicts)} non-free pairs)")
          + f" after {res.oracle_calls} verifier calls, {res.seconds:.1f} s")
    if res.disassembly_feasible is not None:
        print(f"  sequential disassembly: {'feasible' if res.disassembly_feasible else 'not found'}"
              f"; order: {' -> '.join(res.removal_order)}"
              + (f"; stuck: {res.stuck_parts}" if res.stuck_parts else ""))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"case": case, "oracle_version": O.ORACLE_VERSION, "tol": a.tol,
                       "deflection": a.deflection, "accepted": res.accepted,
                       "conflicts": [list(c) for c in res.conflicts], "plan": res.plan,
                       "disassembly_feasible": res.disassembly_feasible,
                       "removal_order": res.removal_order, "history": res.history},
                      f, indent=2)
    return 0 if res.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
