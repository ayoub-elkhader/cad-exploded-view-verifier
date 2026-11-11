# AutoMate examples

Three assemblies copied **unchanged** from the AutoMate dataset:

- the assembly JSON files (`<assemblyId>.json`);
- the STEP files of the parts they use (`step/`).

| Assembly id (prefix) | Parts | Verifier result (`--disassembly`, verifier 2.3) |
|---|---|---|
| `0c4c5dd5…` | 3 | the centroid explosion is accepted directly; straight-line disassembly found |
| `0b7302d4…` | 4 | 1 conflicting pair after the centroid explosion; accepted after one axial-spreading pass; disassembly found |
| `d0582033…` | 8 | 3 conflicting pairs after the centroid explosion; accepted after one axial-spreading pass; disassembly found |

```bash
exploded-view-verify --automate examples/automate/0b7302d45f43f418f0bf1ee1_1b4e887c63cdc9e134906b05_e653cc8c1f0b44f057b8d6a8_default.json \
    --step-dir examples/automate/step --disassembly
```

## Source and license

- **Dataset**: AutoMate, B. Jones, D. Hildreth, D. Chen, I. Baran, V. G. Kim, A. Schulz. *AutoMate: A Dataset and Learning Approach for Automatic Mating of CAD Assemblies.* ACM Transactions on Graphics 40(6), 2021. [doi:10.1145/3478513.3480562](https://doi.org/10.1145/3478513.3480562)
- **Data**: [doi:10.5281/zenodo.7750955](https://doi.org/10.5281/zenodo.7750955) (also on Dryad, [doi:10.5061/dryad.2547d7wvw](https://doi.org/10.5061/dryad.2547d7wvw)). The Zenodo record is released under **Creative Commons Zero v1.0 Universal (CC0)**.
- The assemblies come from public Onshape documents collected by the dataset authors.
- These files are **not** covered by the MIT license of this repository.
