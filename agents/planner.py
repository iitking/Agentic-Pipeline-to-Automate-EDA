
from typing import Any
import json
from workflow.state import EDAState
from workflow.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from workflow.llm_client import LLMClient


class PlannerAgent:

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _generate_heuristic_plan(self, state: EDAState) -> dict[str, Any]:
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
                "task_name": "analyze_relationships",
                "priority": "high",
                "reason": f"Investigate feature associations and hypothesis testing against target variable '{target}'.",
                "parameters": {"target_column": target, "run_hypothesis_tests": True},
            })
            reasoning_points.append(f"Target relationship analysis selected for candidate target '{target}'.")

        return {
            "reasoning": " ".join(reasoning_points),
            "tasks": tasks,
        }

    def run(self, state: EDAState) -> dict[str, Any]:
        profile = state.get("data_profile", {})
        target = state.get("target_column")
        obj = state.get("user_objective", "Complete comprehensive EDA")
        iteration = state.get("iteration", 0)

        profile_summary = (
            f"Dataset '{profile.get('dataset_name', '')}': {profile.get('rows', 0)} rows, "
            f"{profile.get('columns', 0)} columns. Missing rate: {profile.get('missing_rate_pct', 0)}%. "
            f"Duplicates: {profile.get('duplicate_rows', 0)} rows. "
            f"Numeric columns: {len(state.get('schema', {}).get('type_groups', {}).get('numerical', []))}. "
            f"Categorical columns: {len(state.get('schema', {}).get('type_groups', {}).get('categorical', []))}."
        )

        user_prompt = PLANNER_USER_PROMPT.format(
            profile_summary=profile_summary,
            target_column=target or "None specified",
            user_objective=obj,
            iteration=iteration,
        )

        plan_result = self.llm.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_heuristic_plan(state),
        )

        tasks = plan_result.get("tasks", [])
        if not tasks:
            tasks = self._generate_heuristic_plan(state)["tasks"]

        reasoning = plan_result.get("reasoning", "Adaptive plan created based on dataset shape and column types.")

        return {
            "analysis_plan": tasks,
            "plan_reasoning": reasoning,
            "status_message": f"Planning completed: {len(tasks)} analysis tasks scheduled.",
        }
