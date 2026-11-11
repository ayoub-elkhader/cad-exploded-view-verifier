"""Run the deterministic pipeline on a list of assemblies and write one CSV row per case.

    python scripts/run_benchmark.py --source fusion   --root <fusion_root> --list benchmarks/fusion_archive2.csv --out a2.csv
    python scripts/run_benchmark.py --source automate --root <assemblies> --step-dir <steps> --list benchmarks/automate_249.csv --out am.csv
    python scripts/run_benchmark.py --source step     --root <folder>     --list benchmarks/zenodo_vergez.csv --out zen.csv

Each case runs in a child process with a time limit, so one pathological geometry
cannot stop the run. Cases already present in the output file are skipped (resume).
No network or LLM call is made.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

FIELDS = ["case", "status", "parts", "conflicts_centroid", "conflicts_final", "accepted",
          "oracle_calls", "disassembly_feasible", "seconds", "error"]


def load(source, root, case, step_dir):
    if source == "fusion":
        from exploded_view_verifier.datasets import load_fusion
        return load_fusion(os.path.join(root, case, "assembly.json"))[0]
    if source == "automate":
        from exploded_view_verifier.datasets import load_automate
        return load_automate(os.path.join(root, case + ".json"), step_dir)[0]
    from exploded_view_verifier.geometry import load_step_assembly
    return load_step_assembly(os.path.join(root, case))


def run_one(a):
    from exploded_view_verifier.proposers import RandomProposer
    from exploded_view_verifier.repair import explode
    t0 = time.time()
    asm = load(a.source, a.root, a.one, a.step_dir)
    row = {"case": a.one, "parts": len(asm)}
    if len(asm) < 2:
        row.update(status="skipped", error="fewer than 2 parts loaded")
    else:
        res = explode(asm, proposer=RandomProposer(a.seed), check_disassembly=a.disassembly,
                      case_id=a.one)
        row.update(status="done", conflicts_centroid=res.history[0]["conflicts"],
                   conflicts_final=len(res.conflicts), accepted=res.accepted,
                   oracle_calls=res.oracle_calls, disassembly_feasible=res.disassembly_feasible)
    row["seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(row), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["fusion", "automate", "step"])
    ap.add_argument("--root", required=True, help="dataset folder")
    ap.add_argument("--step-dir", help="AutoMate part STEP folder")
    ap.add_argument("--list", help="CSV whose first column is the case id / file name")
    ap.add_argument("--out", help="output CSV")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disassembly", action="store_true")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.one:
        run_one(a)
        return 0
    if not (a.list and a.out):
        ap.error("--list and --out are required")
    if a.source == "automate" and not a.step_dir:
        ap.error("--source automate needs --step-dir")

    with open(a.list, encoding="utf-8") as f:
        cases = [r[0] for r in list(csv.reader(f))[1:] if r]
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {r["case"] for r in csv.DictReader(f)}
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, case in enumerate(cases, 1):
            if case in done:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--source", a.source, "--root", a.root,
                   "--seed", str(a.seed), "--one", case]
            if a.step_dir:
                cmd += ["--step-dir", a.step_dir]
            if a.disassembly:
                cmd.append("--disassembly")
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout,
                                   env={**os.environ, "PYTHONHASHSEED": "0"})
                line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
                row = json.loads(line[7:]) if line else {
                    "case": case, "status": "error",
                    "error": (p.stderr.strip().splitlines() or [f"exit code {p.returncode}"])[-1][:200]}
            except subprocess.TimeoutExpired:
                row = {"case": case, "status": "timeout", "seconds": round(time.time() - t0, 1)}
            w.writerow({k2: row.get(k2, "") for k2 in FIELDS})
            f.flush()
            print(f"[{k}/{len(cases)}] {case}: {row.get('status')} "
                  f"{'accepted' if row.get('accepted') else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
