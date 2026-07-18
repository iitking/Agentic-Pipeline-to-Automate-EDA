"""Statistical Analysis Specialist Agent."""

from typing import Any
from tools.profiling import load_dataset
from tools.statistical_tests import (
    compute_descriptive_stats,
    compute_correlation_matrix,
)
from workflow.state import EDAState


class StatisticsAgent:
    """Agent that performs descriptive summaries, distribution metrics, and correlation analyses."""

    def __init__(self):
        pass

    def run(self, state: EDAState) -> dict[str, Any]:
        """Compute descriptive statistics and correlation matrices."""
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            return {"error": "Missing dataset_path in state"}

        df = load_dataset(dataset_path)
        schema = state.get("schema")

        descriptive = compute_descriptive_stats(df, schema=schema)
        correlations = compute_correlation_matrix(df, schema=schema)

        statistical_results = {
            "descriptive": descriptive,
            "correlations": correlations,
        }

        num_count = len(descriptive.get("numerical", {}))
        cat_count = len(descriptive.get("categorical", {}))

        return {
            "statistical_results": statistical_results,
            "status_message": f"Statistical profiling complete for {num_count} numerical and {cat_count} categorical features.",
        }
