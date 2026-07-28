
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


