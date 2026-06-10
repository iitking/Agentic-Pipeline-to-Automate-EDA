
from typing import Any
import pandas as pd
import numpy as np


def detect_missing_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    findings = []
    n_rows = len(df)
    if n_rows == 0:
        return findings

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            null_pct = round((null_count / n_rows) * 100, 2)
            if null_pct > 50:
                severity = "critical"
                rec = f"Over 50% missing ({null_pct}%). Consider dropping column unless business-critical or using missingness indicator flag."
            elif null_pct > 20:
                severity = "high"
                rec = f"Substantial missingness ({null_pct}%). Consider predictive imputation (KNN/Iterative) or domain-specific default."
            elif null_pct > 5:
                severity = "medium"
                rec = f"Moderate missingness ({null_pct}%). Impute with median for skewed numerical, mean for normal, mode for categorical."
            else:
                severity = "low"
                rec = f"Low missingness ({null_pct}%). Safe to impute with median/mode or drop missing rows."

            findings.append({
                "id": f"DQ-MISSING-{col}",
                "check": "missing_values",
                "column": col,
                "severity": severity,
                "metric": {"null_count": null_count, "null_pct": null_pct},
                "evidence": f"Column '{col}' has {null_count} missing values out of {n_rows} rows ({null_pct}%).",
                "recommendation": rec,
            })

    return findings


