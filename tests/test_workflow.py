"""Integration and workflow tests for the Agentic EDA pipeline."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from tools.safe_executor import SafeExecutor
from workflow.state import create_initial_state
from workflow.graph import run_eda_pipeline
from app.main import app


def test_safe_executor_allowlist_enforcement():
    executor = SafeExecutor()

    # Blocked dangerous or non-allowlisted tool
    res_blocked = executor.execute("os_system", {"command": "ls"})
    assert res_blocked["success"] is False
    assert "not in the approved safe tool allowlist" in res_blocked["error"]

    # Allowed safe tool
    res_allowed = executor.execute("detect_schema", {"df": pytest.importorskip("pandas").DataFrame({"a": [1, 2]})})
    assert res_allowed["success"] is True
    assert "shape" in res_allowed["result"]


def test_initial_state_builder():
    state = create_initial_state("data/customer_churn.csv", target_column="Exited")
    assert state["dataset_path"] == "data/customer_churn.csv"
    assert state["target_column"] == "Exited"
    assert state["iteration"] == 0
    assert state["max_iterations"] == 2


def test_end_to_end_eda_pipeline_execution():
    dataset_file = "data/customer_churn.csv"
    assert Path(dataset_file).exists()

    final_state = run_eda_pipeline(
        dataset_path=dataset_file,
        target_column="Exited",
        user_objective="Analyze churn indicators and evaluate data quality",
        max_iterations=1,
    )

    # Verify pipeline state completion
    assert final_state["data_profile"]["rows"] > 0
    assert "quality_score" in final_state
    assert len(final_state["quality_findings"]) > 0
    assert len(final_state["analysis_plan"]) > 0
    assert len(final_state["validated_insights"]) > 0
    assert len(final_state["chart_paths"]) > 0

    # Verify report generated on disk
    report_path = Path(final_state["final_report_path"])
    assert report_path.exists()
    assert report_path.stat().st_size > 0

    # Verify summary artifacts on disk
    assert Path("artifacts/dataset_profile.json").exists()
    assert Path("artifacts/quality_report.json").exists()
    assert Path("artifacts/summary_statistics.csv").exists()


def test_fastapi_endpoints():
    client = TestClient(app)

    # Health check
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # Analyze endpoint
    resp_analyze = client.post(
        "/api/analyze",
        json={"file_path": "data/customer_churn.csv", "target_column": "Exited"},
    )
    assert resp_analyze.status_code == 200
    job_id = resp_analyze.json()["job_id"]
    assert job_id is not None

    # Status check
    resp_status = client.get(f"/api/status/{job_id}")
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] in ["queued", "processing", "completed"]
