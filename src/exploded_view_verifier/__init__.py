"""Geometric verification and repair of exploded views of CAD assemblies (STEP)."""

from .geometry import load_step_assembly, load_solids, write_step
from .oracle import verify_plan, verify_placed, part_pair_verdict, ORACLE_VERSION
from .proposers import centroid_plan, RandomProposer, ScaleProposer
from .disassembly import sequential_disassembly, simultaneous_feasible
from .repair import explode, RepairResult

__version__ = "0.1.0"

__all__ = [
    "load_step_assembly", "load_solids", "write_step",
    "verify_plan", "verify_placed", "part_pair_verdict", "ORACLE_VERSION",
    "centroid_plan", "RandomProposer", "ScaleProposer",
    "sequential_disassembly", "simultaneous_feasible",
    "explode", "RepairResult",
]
