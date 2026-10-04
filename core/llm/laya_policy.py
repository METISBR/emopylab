"""Laya Policy Client — Tier-2 semantic decision controller for LARC-NSGA3.

Laya (``convaiinnovations/laya``, multilingual) is an open-weights, Apache-2.0
System-1 decision model running locally. It evaluates state narratives and metrics
against typed atomic questions (``choice``) in a single forward pass, returning
calibrated probabilities over admissible evolutionary actions without autoregressive
token generation or JSON syntax brittleness.

This client wraps ``Skills/laya/scripts/laya_client.LayaClient`` and duck-types
``LocalLLMClient.json_call`` so that LARC-NSGA3's control loop, diagnostic traps,
and graceful fallback execute seamlessly.

Two prompt modes are supported (see :func:`build_laya_request`):

* legacy / ``laya_blind`` -- the pre-revision request, byte-for-byte: text
  ``"Search Phase: <narrative> | State: <json>"``, generic action descriptions as
  criteria, client default language (``lang="pt"``);
* ``laya_informed`` -- the payload carries ``laya_text`` (narrative + interpretation,
  numeric features, EMA credit, last three action outcomes), ``laya_criteria``
  (operator-level description with the actual parameters + a "Use when" cue) and
  ``lang="en"``; they are forwarded verbatim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

def laya_skills_dir() -> Optional[Path]:
    """Directory holding ``laya_client.py`` (the Skills/laya wrapper), or None.

    Resolution order (spec 001 FR-008; no user-specific absolute paths):
    1. ``LAYA_SKILLS_DIR`` environment variable (set it on HPC nodes, e.g. SDumont);
    2. ``<workspace>/Skills/laya/scripts`` where <workspace> is two levels above the
       emopylab root (the METISBr/devSuport development layout);
    3. ``<emopylab>/third_party/laya/scripts`` (vendored copy, if shipped).
    """
    env = os.environ.get("LAYA_SKILLS_DIR")
    here = Path(__file__).resolve()
    emopylab_root = here.parents[2]
    candidates = [Path(env)] if env else []
    candidates += [emopylab_root.parents[1] / "Skills" / "laya" / "scripts",
                   emopylab_root / "third_party" / "laya" / "scripts"]
    for cand in candidates:
        if (cand / "laya_client.py").is_file():
            return cand.resolve()
    return None


_SKILLS_LAYA = laya_skills_dir()
if _SKILLS_LAYA is not None and str(_SKILLS_LAYA) not in sys.path:
    sys.path.insert(0, str(_SKILLS_LAYA))

try:
    from laya_client import LayaClient
except ImportError:
    LayaClient = None  # type: ignore

#: Instruction of the closed-set ``choice`` question (unchanged since the first release).
LEGACY_CHOICE_INSTRUCTIONS = (
    "Select the single most effective LARC-NSGA3 search-policy action for the current many-objective search state."
)
_DEFAULT_ACTIONS = ("conv", "div", "var", "ref", "subpop", "pref")


def build_laya_request(
    prompt: Optional[str],
    allowed: Optional[tuple] = None,
    criteria: Optional[Dict[str, str]] = None,
    default_lang: str = "pt",
) -> Dict[str, Any]:
    """Translate a LARC prompt payload into the exact Laya ``choice`` request.

    Returns ``{"mode", "text", "instructions", "criteria", "lang"}`` -- exactly what
    :meth:`LayaPolicyClient.json_call` routes to ``LayaClient.evaluate``. It is pure (no
    model access) so LARC-NSGA3 can log the verbatim request for every Tier-2 decision.

    * Informed mode (payload carries ``laya_text`` and ``laya_criteria``): the text and
      criteria are used verbatim and ``lang`` is taken from the payload (LARC sends "en").
    * Legacy / blind mode (any other payload): identical to the pre-revision client --
      ``"Search Phase: <narrative> | State: <json state>"`` with the payload's
      ``action_descriptions`` as criteria and the client's default language.
    """
    payload: Dict[str, Any] = {}
    if prompt:
        try:
            payload = json.loads(prompt)
        except (TypeError, json.JSONDecodeError):
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if "laya_text" in payload and isinstance(payload.get("laya_criteria"), dict):
        crit = payload["laya_criteria"]
        allowed_list = list(payload.get("allowed_actions") or crit.keys())
        return {
            "mode": "informed",
            "text": str(payload["laya_text"]),
            "instructions": str(payload.get("laya_instructions") or LEGACY_CHOICE_INSTRUCTIONS),
            "criteria": {a: str(crit.get(a, "") or a) for a in allowed_list},
            "lang": str(payload.get("lang") or default_lang),
        }

    state_dict = payload.get("state", {})
    narrative = payload.get("state_narrative", "")
    state_text = f"Search Phase: {narrative} | State: {json.dumps(state_dict)}"

    allowed_list = list(payload.get("allowed_actions") or (allowed or _DEFAULT_ACTIONS))
    desc_map = payload.get("action_descriptions") or (criteria or {})

    criteria_map = {
        a: str(desc_map.get(a, "") or a)
        for a in allowed_list
    }
    return {
        "mode": "legacy",
        "text": state_text,
        "instructions": LEGACY_CHOICE_INSTRUCTIONS,
        "criteria": criteria_map,
        "lang": default_lang,
    }


def _choice_question(instructions: str, criteria: Dict[str, str]) -> Dict[str, Any]:
    if LayaClient is not None:
        return LayaClient.choice(instructions, criteria)
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _jsonable(obj: Any) -> Any:
    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    try:
        return json.loads(json.dumps(obj, default=_default))
    except Exception:
        return str(obj)


class LayaPolicyClient:
    """Decision-model client speaking the LARC-NSGA3 Tier-2 duck-type contract."""

    def __init__(
        self,
        checkpoint: str = "convaiinnovations/laya",
        subfolder: str = "multilingual",
        lang: str = "pt",
        device: Optional[str] = None,
        prefer_local_import: bool = True,
    ) -> None:
        if LayaClient is None:
            raise RuntimeError(
                f"Cannot import LayaClient (searched: {_SKILLS_LAYA or 'LAYA_SKILLS_DIR unset, no default found'}). "
                "Set LAYA_SKILLS_DIR to the directory containing laya_client.py and install the "
                "'laya' package (pip install 'emopylab[laya]')."
            )
        self.client = LayaClient(
            checkpoint=checkpoint,
            subfolder=subfolder,
            device=device,
            prefer_local_import=prefer_local_import,
            default_lang=lang,
        )
        self.lang = lang
        self.checkpoint = checkpoint
        self.subfolder = subfolder
        self.resolved_model = f"{checkpoint} ({subfolder})"
        self.backend_name = "laya-system1"
        self.last_call_status: Dict[str, Any] = {"status": "idle", "detail": ""}

    def json_call(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        allowed: Optional[tuple] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Duck-type json_call expected by LARC-NSGA3's _query_llm_policy.

        The request is built by :func:`build_laya_request`: legacy payloads reproduce the
        pre-revision text/criteria/lang byte-for-byte; informed payloads (``laya_text`` +
        ``laya_criteria``) are forwarded verbatim with the payload's ``lang``. ``system`` is
        accepted for duck-type compatibility only (the ``choice`` primitive has no system
        prompt). The returned dict carries ``raw_answer`` (the model's answer record).
        """
        request = build_laya_request(prompt, allowed=allowed, criteria=criteria, default_lang=self.lang)
        questions = {"action": _choice_question(request["instructions"], request["criteria"])}

        answers = self.client.evaluate(request["text"], questions, lang=request["lang"])
        ans = answers.get("action", {})
        chosen = str(ans.get("choice", "")).strip().lower()
        if not chosen:
            raise RuntimeError("Laya returned an empty choice answer.")

        confidence = float(ans.get("answer_confidence") or ans.get("confidence") or 0.5)
        confidence = max(0.0, min(confidence, 1.0))

        self.last_call_status = {"status": "success", "detail": f"choice={chosen}"}
        return {
            "action": chosen,
            "confidence": confidence,
            "reason_code": f"laya_{chosen}_p{confidence:.2f}",
            "raw_answer": _jsonable(ans),
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.json_call(*args, **kwargs)
