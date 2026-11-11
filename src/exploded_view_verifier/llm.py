"""Optional LLM residual proposer (any OpenAI-compatible chat endpoint).

The model only sees the parts still in conflict (current centre, current vector, bounding
box diagonal) and the conflicting pairs, and answers with new translation vectors for
those parts only. Every answer is checked by the verifier; the LLM never decides whether a
plan is valid.

Configuration (environment variables only, never hard-coded):

* ``LLM_API_KEY``  - API key (required to use this proposer);
* ``LLM_BASE_URL`` - endpoint, default ``https://integrate.api.nvidia.com/v1``;
* ``LLM_MODEL``    - model name, default ``google/gemma-4-31b-it``.

Requires the ``llm`` extra: ``pip install -e .[llm]``. Calls to a hosted endpoint may be
billed by the provider.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from . import geometry as G
from .proposers import ResidualContext

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it"

SYSTEM_PROMPT = (
    "You are a mechanical CAD expert fixing an exploded view. A centroid explosion has "
    "been applied to all parts; MOST parts are already fine. A few parts still overlap. "
    "You will receive ONLY the still-overlapping parts, with their current center, the "
    "translation already applied, their bounding-box diagonal, and the overlapping pairs. "
    "Your job: propose a NEW translation vector [dx,dy,dz] (in mm, relative to the part's "
    "ORIGINAL assembled position, i.e. replacing the current one) for ONLY these parts, so "
    "the listed pairs no longer overlap. Move overlapping parts further apart, and split "
    "coaxial parts along their shared axis with DIFFERENT magnitudes so they don't stay "
    "nested. Respond with ONLY a JSON object mapping each listed part id to [dx,dy,dz]. "
    "No prose, no markdown."
)


def conflict_context(ctx: ResidualContext) -> dict:
    parts = []
    for p in ctx.involved:
        c0, diag = G.center_diag(ctx.assembly[p])
        v = ctx.plan.get(p, [0, 0, 0])
        parts.append({
            "id": p,
            "current_center": [round(c0[k] + v[k], 2) for k in range(3)],
            "current_vector": [round(x, 2) for x in v],
            "bbox_diag": round(diag, 2),
        })
    return {"colliding_parts": parts, "colliding_pairs": [list(x) for x in ctx.conflicts]}


def build_prompt(context: dict, previous_conflicts: Optional[list] = None) -> str:
    msg = "STILL OVERLAPPING:\n" + json.dumps(context, ensure_ascii=False)
    if previous_conflicts:
        msg += (f"\n\nYOUR LAST FIX STILL LEFT {len(previous_conflicts)} OVERLAPPING PAIRS. "
                "Move these parts even further apart, using larger magnitudes and clearly "
                "different directions. Output the corrected JSON (same format).")
    return msg


def parse_vectors(raw: str, allowed_ids: Iterable[str]) -> Tuple[Optional[Dict[str, List[float]]], Optional[str]]:
    """Parse a partial plan from a model answer. Returns (vectors, error)."""
    allowed = set(allowed_ids)
    if not raw:
        return None, "empty answer"
    txt = raw.strip()
    if "```" in txt:
        txt = txt.split("```")[1]
        if txt.lower().startswith("json"):
            txt = txt[4:]
    a, b = txt.find("{"), txt.rfind("}")
    if a == -1 or b == -1:
        return None, "no JSON object found"
    try:
        obj = json.loads(txt[a:b + 1])
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"
    fix = {}
    for k, v in obj.items():
        if k in allowed and isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                fix[k] = [float(v[0]), float(v[1]), float(v[2])]
            except (TypeError, ValueError):
                pass
    if not fix:
        return None, "no usable vector"
    return fix, None


class LLMProposer:
    name = "llm"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.2,
                 timeout: float = 240.0, retries: int = 2):
        key = api_key or os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is not set")
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                             api_key=key, timeout=timeout, max_retries=0)
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.retries = retries
        self.transcript: List[dict] = []

    def _ask(self, messages) -> str:
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=3000)
                msg = resp.choices[0].message
                return msg.content or getattr(msg, "reasoning_content", None) or ""
            except Exception as e:   # network errors, timeouts
                last = e
                if attempt < self.retries:
                    time.sleep(2)
        raise last

    def __call__(self, ctx: ResidualContext):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(conflict_context(ctx),
                                                             ctx.previous_conflicts)}]
        raw = self._ask(messages)
        fix, err = parse_vectors(raw, ctx.involved)
        self.transcript.append({"round": ctx.round, "prompt": messages[1]["content"],
                                "answer": raw, "parse_error": err})
        return fix or {}
