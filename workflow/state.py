"""State definition for the Agentic EDA workflow."""

from typing import TypedDict, Any


class EDAState(TypedDict, total=False):
    """Shared typed state passed between agentic workflow nodes."""

    # Dataset inputs
    dataset_path: str
    dataset_name: str
    target_column: str | None
    user_objective: str

    # Profiling & Schema
    schema: dict[str, Any]
    data_profile: dict[str, Any]

    # Data Quality
    quality_findings: list[dict[str, Any]]
    quality_score: float

    # Planning
    analysis_plan: list[dict[str, Any]]
    plan_reasoning: str

    # Specialist Analyses
    statistical_results: dict[str, Any]
    relationship_results: dict[str, Any]
    hypothesis_test_results: list[dict[str, Any]]

    # Visualizations
    chart_paths: list[str]
    chart_metadata: list[dict[str, Any]]

    # Validation & Insights
    validated_insights: list[dict[str, Any]]
    actionable_recommendations: list[dict[str, Any]]
    warnings: list[str]
    validation_passed: bool
    requires_further_analysis: bool

    # Workflow Control
    iteration: int
    max_iterations: int
    final_report_path: str
    status_message: str
    error: str | None


def create_initial_state(
    dataset_path: str,
    target_column: str | None = None,
    user_objective: str = "Perform comprehensive exploratory data analysis, evaluate quality, and generate insights.",
    max_iterations: int = 2,
) -> EDAState:
    """Create a clean initialized state dict."""
    return {
        "dataset_path": dataset_path,
        "dataset_name": "",
        "target_column": target_column,
        "user_objective": user_objective,
        "schema": {},
        "data_profile": {},
        "quality_findings": [],
        "quality_score": 100.0,
        "analysis_plan": [],
        "plan_reasoning": "",
        "statistical_results": {},
        "relationship_results": {},
        "hypothesis_test_results": [],
        "chart_paths": [],
        "chart_metadata": [],
        "validated_insights": [],
        "actionable_recommendations": [],
        "warnings": [],
        "validation_passed": True,
        "requires_further_analysis": False,
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_report_path": "",
        "status_message": "Initialized",
        "error": None,
    }
