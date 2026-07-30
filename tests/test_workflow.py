
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


