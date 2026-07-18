"""Dataset Understanding Agent."""

from pathlib import Path
from typing import Any
from tools.profiling import profile_dataset, load_dataset, detect_schema
from workflow.state import EDAState


class DatasetAgent:
    """Agent responsible for dataset ingestion, schema inference, and profiling."""

    def __init__(self):
        pass

    def run(self, state: EDAState) -> dict[str, Any]:
        """Profile dataset and populate schema, profile, and candidate target."""
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            return {"error": "Dataset path not provided to DatasetAgent"}

        # Generate profile using deterministic tool
        profile = profile_dataset(dataset_path)
        schema = profile["schema"]

        target_col = state.get("target_column")
        if not target_col and schema.get("inferred_target"):
            inferred = schema["inferred_target"]
            # Auto-adopt candidate target if confidence is high and user did not specify
            if inferred.get("confidence", 0) >= 0.7:
                target_col = inferred.get("column")

        return {
            "dataset_name": Path(dataset_path).name,
            "data_profile": profile,
            "schema": schema,
            "target_column": target_col,
            "status_message": f"Dataset profiled: {profile['rows']} rows, {profile['columns']} columns.",
        }
