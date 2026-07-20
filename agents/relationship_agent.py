"""Relationship Analysis Specialist Agent."""

from typing import Any
import pandas as pd
from tools.profiling import load_dataset
from tools.statistical_tests import run_hypothesis_tests
from workflow.state import EDAState


class RelationshipAgent:
    """Agent that examines feature interactions, target relationships, and hypothesis tests."""

    def __init__(self):
        pass

    def run(self, state: EDAState) -> dict[str, Any]:
        """Perform feature relationship analysis and group comparisons against target."""
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            return {"error": "Missing dataset_path in state"}

        df = load_dataset(dataset_path)
        schema = state.get("schema")
        target = state.get("target_column")

        relationship_results: dict[str, Any] = {
            "target": target,
            "has_target": bool(target and target in df.columns),
            "findings": [],
        }
        hypothesis_tests: list[dict[str, Any]] = []

        if target and target in df.columns:
            target_series = df[target].dropna()
            n_target = len(target_series)
            val_counts = target_series.value_counts()

            # Target distribution metrics
            dist_info = {
                "classes": [str(k) for k in val_counts.index],
                "counts": [int(v) for v in val_counts.values],
                "proportions": [round((v / max(n_target, 1)) * 100, 2) for v in val_counts.values],
                "is_imbalanced": False,
            }

            if len(val_counts) == 2:
                minority_pct = min(dist_info["proportions"])
                dist_info["is_imbalanced"] = minority_pct < 20.0
                if dist_info["is_imbalanced"]:
                    relationship_results["findings"].append({
                        "type": "class_imbalance",
                        "severity": "high",
                        "evidence": f"Target '{target}' has significant class imbalance: minority class is {minority_pct}%.",
                        "caveat": "Standard accuracy will be misleading; precision-recall AUC, SMOTE, or class weights recommended.",
                    })

            relationship_results["target_distribution"] = dist_info

            # Run hypothesis tests between features and target
            hypothesis_tests = run_hypothesis_tests(df, target_column=target, schema=schema)

            # Summarize significant tests
            for test in hypothesis_tests:
                if test.get("is_significant"):
                    feat = test.get("feature") or test.get("feature_a")
                    p_val = test.get("p_value")
                    eff = test.get("effect_label", "moderate")
                    relationship_results["findings"].append({
                        "type": "significant_association",
                        "feature": feat,
                        "p_value": p_val,
                        "evidence": f"Feature '{feat}' exhibits statistically significant difference across '{target}' groups (p={p_val}, effect={eff}).",
                        "caveat": "Correlation/association does not imply causality; unobserved confounders may exist.",
                    })

        return {
            "relationship_results": relationship_results,
            "hypothesis_test_results": hypothesis_tests,
            "status_message": f"Relationship analysis complete: {len(hypothesis_tests)} hypothesis tests evaluated.",
        }
