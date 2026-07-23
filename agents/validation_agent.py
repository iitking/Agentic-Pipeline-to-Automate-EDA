"""Validation Agent auditing claims against ground truth analytics."""

import re
from typing import Any
from workflow.state import EDAState
from workflow.prompts import VALIDATION_SYSTEM_PROMPT, VALIDATION_USER_PROMPT
from workflow.llm_client import LLMClient


class ValidationAgent:
    """Audits insights and claims against calculated ground truth values to prevent hallucination and false causality."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _sanitize_causal_language(self, text: str) -> tuple[str, bool]:
        """Detect and sanitize unjustified causal language into associative language."""
        causal_replacements = [
            (r"\bcauses\b", "is associated with"),
            (r"\bcaused\b", "was associated with"),
            (r"\bleads to\b", "correlates with"),
            (r"\bdrives\b", "is linked to"),
            (r"\bresults in\b", "co-occurs with"),
            (r"\bproves that\b", "suggests that"),
        ]
        modified = text
        had_causal = False
        for pattern, replacement in causal_replacements:
            if re.search(pattern, modified, re.IGNORECASE):
                had_causal = True
                modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        return modified, had_causal

    def _validate_heuristic(self, state: EDAState) -> dict[str, Any]:
        """Rule-based verification of numerical metrics and claim sanity."""
        insights = state.get("validated_insights", [])
        warnings = list(state.get("warnings", []))
        discrepancies = []

        validated_insights = []
        for item in insights:
            obs = item.get("observation", "")
            clean_obs, had_causal = self._sanitize_causal_language(obs)
            if had_causal:
                warnings.append(f"Causal phrasing in insight '{item.get('headline', '')}' sanitized to associative terminology.")

            item_copy = dict(item)
            item_copy["observation"] = clean_obs
            validated_insights.append(item_copy)

        # Check iteration termination
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 2)

        # Only require further analysis if quality score is very low and iteration == 0
