# Datasets

The thesis evaluated the verifier on four public sources. This repository ships the **loaders**, the **lists of assemblies** used (`benchmarks/`) and a **script to run the pipeline on them** (`scripts/run_benchmark.py`). The data itself must be downloaded from the official sources, except for a few CC0 / CC BY examples in `examples/`.

| Source | License | In this repository | Loader |
|---|---|---|---|
| Fusion 360 Gallery Assembly | Autodesk research license: **non-commercial research only**, no redistribution of the whole dataset | identifier lists only | `datasets.load_fusion` / `--fusion` |
| AutoMate (Onshape) | **CC0** (Zenodo record) | identifier list + 3 assemblies in `examples/automate/` | `datasets.load_automate` / `--automate` |
| Vergez et al. reassembly set (Zenodo) | **CC BY 4.0** | file list + 1 assembly in `examples/zenodo/` | plain STEP |
| Your own STEP files | yours | — | `load_step_assembly` / positional argument |

## Fusion 360 Gallery Assembly

- **Citation**: K. D. D. Willis et al. *JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints.* CVPR 2022.
- **Download**: [github.com/AutodeskAILab/Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset), Assembly dataset. Accepting its license is required, and use is limited to non-commercial research.
- **Layout expected by the loader**: one folder per assembly, containing `assembly.json` and the `<body>.step` files it references.
- **Units**: Fusion stores geometry in centimetres. The loader converts translations with `scale=10`, which matches the STEP files in millimetres used here.

```bash
exploded-view-verify --fusion <root>/<assembly_id>/assembly.json --disassembly
```

**Lists**: `benchmarks/fusion_archive1.csv` and `benchmarks/fusion_archive2.csv`, 751 assemblies each, from two different download archives of the dataset. One assembly per archive crashed the original runs and was excluded from the thesis results: `57096_6d459f9d` (archive 1) and `135847_852f787d` (archive 2).

## AutoMate

- **Citation**: B. Jones, D. Hildreth, D. Chen, I. Baran, V. G. Kim, A. Schulz. *AutoMate: A Dataset and Learning Approach for Automatic Mating of CAD Assemblies.* ACM TOG 40(6), 2021.
- **Download**: Zenodo [doi:10.5281/zenodo.7750955](https://doi.org/10.5281/zenodo.7750955), CC0. You need `assemblies.zip` (one JSON per assembly) and `step.zip` (one STEP per part; 13 GB, but only the parts of the listed assemblies are required).
- **Units**: transforms are 4×4 row-major with translations in metres (`scale=1000`); STEP geometry is in millimetres.

```bash
exploded-view-verify --automate <assemblies>/<assembly_id>.json --step-dir <step_folder> --disassembly
```

**List**: `benchmarks/automate_249.csv`, 249 assemblies drawn by part count (3–30 parts) from the assemblies with complete STEP geometry and at least one mate.

## Vergez et al. reassembly set

- **Citation**: L. Vergez, A. Polette, J.-P. Pernot, Zenodo [doi:10.5281/zenodo.10057280](https://doi.org/10.5281/zenodo.10057280), CC BY 4.0. The thesis used version 2.
- **List**: `benchmarks/zenodo_vergez.csv`, 145 STEP files. The names flatten the dataset paths: `/` becomes `__`.

## Running the pipeline on a list

```bash
python scripts/run_benchmark.py --source fusion   --root <fusion_root>   --list benchmarks/fusion_archive2.csv --out results_a2.csv
python scripts/run_benchmark.py --source automate --root <assemblies>    --step-dir <step_folder> --list benchmarks/automate_249.csv --out results_automate.csv
python scripts/run_benchmark.py --source step     --root <zenodo_folder> --list benchmarks/zenodo_vergez.csv --out results_zenodo.csv
```

For each assembly the script writes:
- the number of loaded parts;
- the conflicts after the centroid explosion and at the end;
- whether the view was accepted;
- the verifier calls and the time.

With `--disassembly`, it also writes the disassembly result. It resumes where it stopped, so cases already in the output file are skipped. Only the deterministic pipeline (centroid + axial spreading + seeded random proposer) is available offline. The LLM proposer needs an API key and may be billed.

These lists let anyone re-run the pipeline on the same assemblies. The numbers in the thesis also depend on the verifier version and the exact settings used at the time, so this repository makes no claim that it reproduces them.
