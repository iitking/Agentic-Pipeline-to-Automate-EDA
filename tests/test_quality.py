
import pytest
import pandas as pd
import numpy as np
from tools.data_quality import (
    detect_missing_values,
    detect_duplicates,
    detect_numerical_outliers,
    detect_schema_anomalies,
    calculate_data_quality_score,
    run_quality_audit,
)
from tools.profiling import detect_schema


def test_detect_missing_values():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "C": [np.nan, np.nan, np.nan, np.nan, 1],
    })
    findings = detect_missing_values(df)
    assert len(findings) == 2
    cols = [f["column"] for f in findings]
    assert "A" in cols
    assert "C" in cols
    
    c_finding = next(f for f in findings if f["column"] == "C")
    assert c_finding["severity"] == "critical"  # 80% missing


