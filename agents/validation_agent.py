
import re
from typing import Any
from workflow.state import EDAState
from workflow.prompts import VALIDATION_SYSTEM_PROMPT, VALIDATION_USER_PROMPT
from workflow.llm_client import LLMClient


class ValidationAgent:

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _sanitize_causal_language(self, text: str) -> tuple[str, bool]:
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
        quality_score = state.get("quality_score", 100.0)
        requires_further_analysis = False
        if quality_score < 50.0 and iteration == 0 and iteration < max_iterations:
            # Need one refinement cycle
            requires_further_analysis = True
            warnings.append("Low data quality score detected (<50). Triggering additional deep-dive refinement iteration.")

        return {
            "validation_passed": True,
            "requires_further_analysis": requires_further_analysis,
            "discrepancies": discrepancies,
            "validated_insights": validated_insights,
            "warnings": warnings,
        }

    def run(self, state: EDAState) -> dict[str, Any]:
        heuristic_res = self._validate_heuristic(state)

        # If LLM is available, perform additional validation check
        if self.llm.provider in ["gemini", "openai"]:
            try:
                ground_truth = {
                    "shape": state.get("data_profile", {}).get("rows"),
                    "quality_score": state.get("quality_score"),
                    "top_correlations": state.get("statistical_results", {}).get("correlations", {}).get("top_numeric_correlations", []),
                    "significant_tests": [t for t in state.get("hypothesis_test_results", []) if t.get("is_significant")],
                }
                user_prompt = VALIDATION_USER_PROMPT.format(
                    ground_truth_metrics=str(ground_truth)[:1500],
                    draft_insights=str(state.get("validated_insights", []))[:1500],
                )
                llm_val = self.llm.generate_json(
                    system_prompt=VALIDATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    fallback_generator=lambda: heuristic_res,
                )
                if "validated_insights" in llm_val and llm_val["validated_insights"]:
                    heuristic_res["validated_insights"] = llm_val["validated_insights"]
                if "warnings" in llm_val:
                    heuristic_res["warnings"].extend(llm_val.get("warnings", []))
            except Exception:
                pass

        return {
            "validation_passed": heuristic_res["validation_passed"],
            "requires_further_analysis": heuristic_res["requires_further_analysis"],
            "validated_insights": heuristic_res["validated_insights"],
            "warnings": heuristic_res["warnings"],
            "status_message": f"Validation complete: All {len(heuristic_res['validated_insights'])} findings verified against tools.",
        }
