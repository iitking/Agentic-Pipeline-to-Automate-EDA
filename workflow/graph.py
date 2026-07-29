
from pathlib import Path
from typing import Any, Literal
from langgraph.graph import StateGraph, END

from workflow.state import EDAState, create_initial_state
from agents.dataset_agent import DatasetAgent
from agents.planner import PlannerAgent
from agents.quality_agent import QualityAgent
from agents.statistics_agent import StatisticsAgent
from agents.visualization_agent import VisualizationAgent
from agents.relationship_agent import RelationshipAgent
from agents.insight_agent import InsightAgent
from agents.validation_agent import ValidationAgent
from reporting.report_generator import ReportGenerator


# Initialize agent instances
dataset_agent = DatasetAgent()
planner_agent = PlannerAgent()
quality_agent = QualityAgent()
statistics_agent = StatisticsAgent()
visualization_agent = VisualizationAgent()
relationship_agent = RelationshipAgent()
insight_agent = InsightAgent()
validation_agent = ValidationAgent()
report_generator = ReportGenerator()


# Node functions
def profile_dataset_node(state: EDAState) -> dict[str, Any]:
    return dataset_agent.run(state)


def plan_analysis_node(state: EDAState) -> dict[str, Any]:
    return planner_agent.run(state)


def run_quality_checks_node(state: EDAState) -> dict[str, Any]:
    return quality_agent.run(state)


def run_statistical_analysis_node(state: EDAState) -> dict[str, Any]:
    return statistics_agent.run(state)


def generate_visualizations_node(state: EDAState) -> dict[str, Any]:
    return visualization_agent.run(state)


def analyze_relationships_node(state: EDAState) -> dict[str, Any]:
    return relationship_agent.run(state)


def synthesize_insights_node(state: EDAState) -> dict[str, Any]:
    return insight_agent.run(state)


def validate_findings_node(state: EDAState) -> dict[str, Any]:
    return validation_agent.run(state)


def generate_report_node(state: EDAState) -> dict[str, Any]:
    report_path = report_generator.generate_report(state)
    return {
        "final_report_path": report_path,
        "status_message": f"EDA complete! Report saved at {report_path}",
    }


def should_continue_condition(state: EDAState) -> Literal["plan_analysis", "generate_report"]:
    requires_more = state.get("requires_further_analysis", False)
    current_iter = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if requires_more and current_iter < max_iter - 1:
        # Increment iteration count
        state["iteration"] = current_iter + 1
        return "plan_analysis"

    return "generate_report"


def build_eda_graph() -> StateGraph:
    workflow = StateGraph(EDAState)

    # Register nodes
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("plan_analysis", plan_analysis_node)
    workflow.add_node("run_quality_checks", run_quality_checks_node)
    workflow.add_node("run_statistical_analysis", run_statistical_analysis_node)
    workflow.add_node("generate_visualizations", generate_visualizations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("synthesize_insights", synthesize_insights_node)
    workflow.add_node("validate_findings", validate_findings_node)
    workflow.add_node("generate_report", generate_report_node)

    # Define edges
    workflow.set_entry_point("profile_dataset")
    workflow.add_edge("profile_dataset", "plan_analysis")
    workflow.add_edge("plan_analysis", "run_quality_checks")
    workflow.add_edge("run_quality_checks", "run_statistical_analysis")
    workflow.add_edge("run_statistical_analysis", "generate_visualizations")
    workflow.add_edge("generate_visualizations", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "synthesize_insights")
    workflow.add_edge("synthesize_insights", "validate_findings")

    # Conditional feedback loop edge
    workflow.add_conditional_edges(
        "validate_findings",
        should_continue_condition,
        {
            "plan_analysis": "plan_analysis",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()


def run_eda_pipeline(
    dataset_path: str,
    target_column: str | None = None,
    user_objective: str = "Perform comprehensive exploratory data analysis, evaluate quality, and generate insights.",
    max_iterations: int = 2,
) -> EDAState:
    initial_state = create_initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        user_objective=user_objective,
        max_iterations=max_iterations,
    )

    graph = build_eda_graph()
    final_state = graph.invoke(initial_state)
    return final_state
