"""Visualization Specialist Agent."""

from pathlib import Path
from typing import Any
from tools.profiling import load_dataset
from tools.chart_generator import generate_eda_charts
from workflow.state import EDAState


class VisualizationAgent:
    """Agent that generates interactive, publication-quality visualizations."""

    def __init__(self, output_dir: str | Path = "artifacts/charts"):
        self.output_dir = Path(output_dir)

    def run(self, state: EDAState) -> dict[str, Any]:
        """Generate charts based on dataset profile and target variable."""
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            return {"error": "Missing dataset_path in state"}

        df = load_dataset(dataset_path)
        schema = state.get("schema")
        target_col = state.get("target_column")

        charts = generate_eda_charts(
            df=df,
            output_dir=self.output_dir,
            target_column=target_col,
            schema=schema,
        )

        chart_paths = [c["file_path"] for c in charts]

        return {
            "chart_paths": chart_paths,
            "chart_metadata": charts,
            "status_message": f"Generated {len(charts)} interactive visual artifacts.",
        }
