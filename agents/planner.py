"""Planning Agent responsible for dynamic task selection and workflow orchestration."""

from typing import Any
import json
from workflow.state import EDAState
from workflow.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from workflow.llm_client import LLMClient


class PlannerAgent:
    """Creates a structured, adaptive analysis plan based on dataset profile and goals."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _generate_heuristic_plan(self, state: EDAState) -> dict[str, Any]:
        """Generate deterministic, expert analysis plan tailored to dataset characteristics."""
        profile = state.get("data_profile", {})
        schema = state.get("schema", {})
        target = state.get("target_column")
        iteration = state.get("iteration", 0)

        tasks = []
        reasoning_points = []

        # 1. Quality Audit
        tasks.append({
            "task_name": "run_quality_checks",
            "priority": "high",
            "reason": f"Audit dataset for missing values ({profile.get('missing_rate_pct', 0)}%), duplicates, outliers, and schema anomalies.",
            "parameters": {"check_types": ["missing_values", "duplicates", "outliers", "schema_anomalies"]},
        })
        reasoning_points.append("Data quality audit selected to establish data health baseline.")

        # 2. Statistical Analysis
        tasks.append({
            "task_name": "run_statistical_analysis",
            "priority": "high",
            "reason": "Calculate descriptive statistics (means, medians, quantiles, skewness) and correlation matrices.",
            "parameters": {"methods": ["descriptive_stats", "pearson_spearman_correlation"]},
        })
        reasoning_points.append("Statistical calculations selected for distributional and correlational properties.")

        # 3. Visualizations
        tasks.append({
            "task_name": "generate_visualizations",
            "priority": "medium",
            "reason": "Render interactive distributions, heatmaps, and categorical frequency plots.",
            "parameters": {"include_heatmap": True, "max_distribution_plots": 4},
        })
        reasoning_points.append("Visualizations selected for distribution patterns and outliers.")

        # 4. Target Relationship Analysis if target exists
        if target:
            tasks.append({
