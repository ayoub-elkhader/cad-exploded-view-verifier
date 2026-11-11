# cad-exploded-view-verifier

**Status: research prototype.** Geometric verification and repair of exploded views of CAD assemblies (STEP), built on OpenCASCADE (pythonocc-core).

An exploded view moves every part of an assembly away from the others. A simple radial explosion, or one proposed by a language model, often leaves parts inside each other (a shaft still inside its bearings, a pin still in its hole). This library checks an exploded view against the exact geometry, says *which* pairs of parts still interfere, and repairs the view in a propose → verify → repair loop. When the check itself cannot conclude, it **abstains** instead of accepting the view.

It is the core of my master's research on exploded views of CAD assemblies with language models, whose central argument is that the geometric verifier, not the proposer (language model or not), is what makes the final views interference-free.

## How it works

```
STEP file ──► parts (one solid = one part)
                │
                ▼
     centroid explosion (propose)
                │
                ▼
     interference verifier ──── accepted ──► plan + optional disassembly order
                │ conflicts
                ▼
     axial spreading of colliding clusters (deterministic)
                │ still conflicts
                ▼
     residual proposer for the conflicting parts only
     (random | scale | optional LLM), each proposal re-verified
```

**Interference verifier** (`oracle.py`). For each pair of parts it returns `free`, `exact_fallback` (boundaries within the tolerance: contact or interpenetration), `inclusion` (one solid fully inside the other) or `undetermined`. Face pairs are pre-filtered conservatively on exact face boxes and triangle boxes. The survivors go through the exact B-rep distance (`BRepExtrema_DistShapeShape`). An exact distance that contradicts the meshes is not trusted: the face/face section decides it, and failing that a bracket on fine meshes. Inclusion is found with a point-in-solid test. Undetermined pairs are reported as conflicts. The default tolerance is 0.1 mm on meshes with a 0.5 mm deflection. The module docstring lists the variants (2.2, 2.3 by default, 2.3.1, 2.4) and the reasons behind each one.

**Disassembly** (`disassembly.py`). A view free of interference in its final state is not necessarily a disassembly. Two models are provided:
- straight-line *simultaneous* motion;
- *sequential* removal along the plan vectors. A part leaves only if it creates no new contact during its travel.

A feasibility repair tries alternative extraction directions for the stuck parts.

**Proposers** (`proposers.py`, `llm.py`): centroid, axial spreading (principal axis of the colliding cluster), seeded random directions, scale-up. There is also an optional LLM proposer that works with any OpenAI-compatible endpoint. The LLM only sees the conflicting parts, and every answer is verified.

## Installation

`pythonocc-core` is distributed through conda-forge only.

```bash
conda env create -f environment.yml
conda activate xview
```

To use the optional LLM proposer, run `pip install -e .[llm]`. Then set `LLM_API_KEY`, and optionally `LLM_BASE_URL` and `LLM_MODEL`, in your environment (see `.env.example`). Calls to a hosted endpoint may be billed by the provider.

## Usage

```bash
# built-in synthetic assemblies (generated from OCCT primitives)
exploded-view-verify --sample shaft_stack --disassembly
exploded-view-verify --write-samples examples/

# your own STEP file (every solid is one part)
exploded-view-verify my_assembly.step --disassembly --out plan.json
exploded-view-verify my_assembly.step --check plan.json        # verify an existing plan
exploded-view-verify my_assembly.step --proposer llm           # optional, needs LLM_API_KEY
```

```python
from exploded_view_verifier import load_step_assembly, explode, verify_plan

asm = load_step_assembly("my_assembly.step")
res = explode(asm, check_disassembly=True)
print(res.accepted, res.conflicts, res.removal_order)
details = {}
verify_plan(asm, res.plan, details=details)     # {(part_a, part_b): verdict} for non-free pairs
```

## Demo (output of the commands above, verifier 2.3)

| Sample | Parts | What happens | Result |
|---|---|---|---|
| `pin_in_plate` | 2 | the centroid explosion already separates the pin from the plate | accepted, disassembly order `pin -> plate` |
| `shaft_stack` | 5 | the centroid explosion leaves 1 conflicting pair; one axial-spreading pass removes it | accepted after 2 verifier calls, disassembly found |
| `caged_cube` | 2 | a cube floating in a closed box: no interference, but no straight-line way out | accepted, **no disassembly found** (both parts stuck) |

## Tests

```bash
pytest
```

44 tests pass locally (Windows 10, Python 3.10, pythonocc-core 7.9.0). They cover:
- 27 verifier tests:
  - cases with a known mathematical answer: cubes at given gaps, nested cubes, a hollow box, cylinders in holes with 1 mm / 0.05 mm clearance or a press fit, nested spheres, a torus, plates tangent to curved faces between mesh nodes, untriangulated faces;
  - abstention when every exact distance fails;
- the triangle–triangle distance against dense sampling;
- the repair loop, the proposers, the disassembly test, the CLI and the parsing of LLM answers (without any network call).

CI runs the same tests on Python 3.10–3.12 with conda-forge.

## Limitations

- **Rigid translations only.** Rotations, screw motions, flexible parts and fasteners are not modelled.
- **Discrete checks.** The disassembly test samples positions along each path, so it does not prove continuous clearance between two samples. "No order found" means that no straight-line removal exists along the tried directions; it does not prove that disassembly is impossible.
- **The verifier is conservative, not certified.** It depends on OCCT and on meshes of a given deflection. Its known failure modes are handled by abstention, but abstention can reject valid views.
- **Simple STEP loading.** Every solid becomes one part: product structure, names and materials are not read.
- **Cost.** Exact distances are expensive, and large assemblies (hundreds of parts) can take minutes per check.
- **No datasets in this repository.** The thesis evaluated the method on public datasets (Fusion 360 Gallery Assembly, AutoMate) that cannot be redistributed here, so those experiments are not reproduced in this repository.

## Related repositories

- [cad-exploded-view-studio](https://github.com/ayoub-elkhader/cad-exploded-view-studio): local 3D viewer built on this library.
- [cad-assembly-assistant](https://github.com/ayoub-elkhader/cad-assembly-assistant): question answering on assemblies, grounded in facts computed by this library.
- [step-exploded-view-api](https://github.com/ayoub-elkhader/step-exploded-view-api): the first, unverified radial approach, as a REST API.

## License

MIT, see [LICENSE](LICENSE).
