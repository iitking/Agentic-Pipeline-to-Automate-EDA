"""Unit tests for data quality auditing module."""

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


def test_detect_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 3, 1],
        "val": ["x", "y", "z", "x"],
    })
    findings = detect_duplicates(df)
    assert len(findings) >= 1
    dup_row_finding = next(f for f in findings if f["id"] == "DQ-DUP-ROWS")
    assert dup_row_finding["metric"]["duplicate_count"] == 1


def test_detect_numerical_outliers():
    # 20 points, one extreme outlier
    data = [10.0] * 19 + [1000.0]
    df = pd.DataFrame({"score": data})
    findings = detect_numerical_outliers(df)
    assert len(findings) == 1
    assert findings[0]["column"] == "score"
    assert findings[0]["metric"]["outlier_count"] == 1


def test_quality_score_calculation():
    findings = [
        {"severity": "critical"},
        {"severity": "medium"},
    ]
    score = calculate_data_quality_score(findings, 100, 5)
    # 100 - 15 - 3 = 82.0
    assert score == 82.0


def test_run_quality_audit_integration():
    df = pd.read_csv("data/customer_churn.csv")
    result = run_quality_audit(df)
    assert "quality_score" in result
    assert "summary" in result
    assert "findings" in result
    assert isinstance(result["findings"], list)
    assert 0 <= result["quality_score"] <= 100
