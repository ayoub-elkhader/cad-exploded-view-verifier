"""Loaders for public CAD assembly datasets.

No dataset is bundled with this package. The repository has a few AutoMate assemblies in
``examples/automate`` (CC0) and one Zenodo file in ``examples/zenodo`` (CC BY 4.0). Download the data from the official sources (see
``DATASETS.md``) and point these loaders to it.

* **Fusion 360 Gallery Assembly** (Willis et al., 2022): one folder per assembly with
  ``assembly.json`` and one STEP file per body. Occurrence transforms are composed along
  the assembly tree; translations are in centimetres (``scale=10`` gives millimetres).
* **AutoMate** (Jones et al., 2021): one JSON per assembly (``parts``, ``occurrences``
  with a 4x4 row-major transform) and one STEP file per part in a shared folder.
  Translations are in metres (``scale=1000``); STEP geometry is already in millimetres.

Both return an assembly ``{part_id: [placed, meshed solids]}`` usable by every function of
this package, plus a ``names`` dict when names are available.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from . import geometry as G


# ----------------------------------------------------------------------------- Fusion 360
def _fusion_body_ids(data, occ_id):
    occ = data["occurrences"][occ_id]
    if isinstance(occ.get("bodies"), dict) and occ["bodies"]:
        return list(occ["bodies"].keys())
    comp = (data.get("components") or {}).get(occ.get("component"), {})
    b = comp.get("bodies", [])
    return list(b) if isinstance(b, list) else []


def _fusion_body_path(data, folder, body_id):
    step = ((data.get("bodies") or {}).get(body_id) or {}).get("step")
    return os.path.join(folder, step) if step else None


def _fusion_parent_map(data):
    parent = {}

    def walk(node, par):
        for oid, children in (node or {}).items():
            parent[oid] = par
            if isinstance(children, dict) and children:
                walk(children, oid)

    walk((data.get("tree") or {}).get("root", {}), None)
    return parent


def _fusion_trsf(tr, scale):
    from OCC.Core.gp import gp_Trsf
    o, xa, ya, za = tr["origin"], tr["x_axis"], tr["y_axis"], tr["z_axis"]
    t = gp_Trsf()
    t.SetValues(xa["x"], ya["x"], za["x"], o["x"] * scale,
                xa["y"], ya["y"], za["y"], o["y"] * scale,
                xa["z"], ya["z"], za["z"], o["z"] * scale)
    return t


def _fusion_world_trsf(data, occ_id, scale, parents, cache):
    """World transform of an occurrence: parent transforms composed root -> leaf."""
    from OCC.Core.gp import gp_Trsf
    if occ_id in cache:
        return cache[occ_id]
    tr = ((data.get("occurrences") or {}).get(occ_id) or {}).get("transform")
    local = _fusion_trsf(tr, scale) if tr else gp_Trsf()
    par = parents.get(occ_id)
    world = local if not par else _fusion_world_trsf(data, par, scale, parents, cache).Multiplied(local)
    cache[occ_id] = world
    return world


def load_fusion(assembly_json: str, scale: float = 10.0,
                deflection: float = G.DEFLECTION) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Load a Fusion 360 Gallery assembly. Returns (assembly, names)."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    folder = os.path.dirname(os.path.abspath(assembly_json))
    with open(assembly_json, encoding="utf-8") as f:
        data = json.load(f)
    parents, cache = _fusion_parent_map(data), {}
    occ, names = {}, {}
    for oid, o in (data.get("occurrences") or {}).items():
        shapes = []
        for bid in _fusion_body_ids(data, oid):
            p = _fusion_body_path(data, folder, bid)
            if not p or not os.path.exists(p):
                continue
            t = _fusion_world_trsf(data, oid, scale, parents, cache)
            shapes += [G.mesh(BRepBuilderAPI_Transform(s, t, True).Shape(), deflection)
                       for s in G.load_solids(p)]
        if shapes:
            occ[oid] = shapes
            names[oid] = o.get("name", oid)
    # "Bodies-only" assemblies: bodies are already in world coordinates and not all of
    # them are referenced by occurrences. Use them if that yields more parts.
    if len(occ) < 2:
        bodies, bnames = {}, {}
        for bid, b in (data.get("bodies") or {}).items():
            p = _fusion_body_path(data, folder, bid) or os.path.join(folder, f"{bid}.step")
            if not os.path.exists(p):
                continue
            sols = [G.mesh(s, deflection) for s in G.load_solids(p)]
            if sols:
                bodies[bid] = sols
                bnames[bid] = (b or {}).get("name", bid)
        if len(bodies) > len(occ):
            occ, names = bodies, bnames
    return occ, names


# ----------------------------------------------------------------------------- AutoMate
def _automate_trsf(m, scale):
    """gp_Trsf from a row-major 4x4 matrix; only the translation is scaled (m -> mm)."""
    from OCC.Core.gp import gp_Trsf
    t = gp_Trsf()
    t.SetValues(m[0], m[1], m[2], scale * m[3],
                m[4], m[5], m[6], scale * m[7],
                m[8], m[9], m[10], scale * m[11])
    return t


def load_automate(assembly_json: str, step_dir: str, scale: float = 1000.0,
                  deflection: float = G.DEFLECTION) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Load an AutoMate assembly. Hidden occurrences and parts without STEP are skipped.
    Returns (assembly, names); part keys are occurrence ids."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    with open(assembly_json, encoding="utf-8") as f:
        asm = json.load(f)
    parts = asm.get("parts", [])
    occ, names = {}, {}
    for o in asm.get("occurrences", []):
        k = o.get("part")
        if o.get("hidden") or k is None or not 0 <= k < len(parts) or not parts[k].get("has_step"):
            continue
        m = o.get("transform")
        path = os.path.join(step_dir, parts[k]["id"] + ".step")
        if not m or len(m) < 12 or not os.path.exists(path):
            continue
        t = _automate_trsf(m, scale)
        sols = [G.mesh(BRepBuilderAPI_Transform(s, t, True).Shape(), deflection)
                for s in G.load_solids(path)]
        if sols:
            occ[o["id"]] = sols
            names[o["id"]] = parts[k].get("name") or o.get("name") or o["id"]
    return occ, names


def automate_part_ids(assembly_json: str):
    """STEP part ids used by the visible occurrences of an AutoMate assembly."""
    with open(assembly_json, encoding="utf-8") as f:
        asm = json.load(f)
    parts = asm.get("parts", [])
    return sorted({parts[o["part"]]["id"] for o in asm.get("occurrences", [])
                   if not o.get("hidden") and o.get("part") is not None
                   and 0 <= o["part"] < len(parts) and parts[o["part"]].get("has_step")})
