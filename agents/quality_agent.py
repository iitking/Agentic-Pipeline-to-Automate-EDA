"""Data Quality Specialist Agent."""

from typing import Any
from tools.profiling import load_dataset
from tools.data_quality import run_quality_audit
from workflow.state import EDAState


class QualityAgent:
    """Agent that audits missing values, duplicates, outliers, and schema anomalies."""

    def __init__(self):
        pass

    def run(self, state: EDAState) -> dict[str, Any]:
        """Perform comprehensive data quality audit using deterministic tools."""
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            return {"error": "Missing dataset_path in state"}

        df = load_dataset(dataset_path)
        schema = state.get("schema")

        audit_results = run_quality_audit(df, schema=schema)

        return {
            "quality_findings": audit_results["findings"],
            "quality_score": audit_results["quality_score"],
            "status_message": f"Quality audit complete: {len(audit_results['findings'])} issues flagged. Health score: {audit_results['quality_score']}/100.",
        }
