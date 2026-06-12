
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



def detect_duplicates(df: pd.DataFrame, schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    findings = []
    n_rows = len(df)
    if n_rows == 0:
        return findings

    # Exact duplicate rows
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        dup_pct = round((dup_rows / n_rows) * 100, 2)
        severity = "high" if dup_pct > 5 else "medium"
        findings.append({
            "id": "DQ-DUP-ROWS",
            "check": "duplicates",
            "column": "all",
            "severity": severity,
            "metric": {"duplicate_count": dup_rows, "duplicate_pct": dup_pct},
            "evidence": f"Found {dup_rows} exact duplicate rows ({dup_pct}% of total records).",
            "recommendation": "Deduplicate dataset by keeping first occurrence or investigate data collection pipeline duplication.",
        })

    # Duplicate IDs if identifier column exists
    if schema and "type_groups" in schema:
        id_cols = schema["type_groups"].get("identifier", [])
        for id_col in id_cols:
            if id_col in df.columns:
                dup_ids = int(df[id_col].duplicated().sum())
                if dup_ids > 0:
                    findings.append({
                        "id": f"DQ-DUP-ID-{id_col}",
                        "check": "duplicate_identifiers",
                        "column": id_col,
                        "severity": "critical",
                        "metric": {"duplicate_ids": dup_ids},
                        "evidence": f"Identifier column '{id_col}' contains {dup_ids} duplicate values.",
                        "recommendation": "Investigate entity resolution. An identifier should be strictly unique per record.",
                    })

    return findings



def detect_numerical_outliers(df: pd.DataFrame, schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    findings = []
    n_rows = len(df)
    if n_rows < 10:
        return findings

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if schema and "type_groups" in schema:
        # Exclude identifiers and booleans
        id_cols = set(schema["type_groups"].get("identifier", []))
        bool_cols = set(schema["type_groups"].get("boolean", []))
        numeric_cols = [c for c in numeric_cols if c not in id_cols and c not in bool_cols]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10 or series.nunique() <= 1 or set(series.unique()).issubset({0, 1}):
            continue

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        if iqr == 0:
            std = float(series.std())
            if std == 0:
                continue
            mean = float(series.mean())
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            outliers = series[(series < lower_bound) | (series > upper_bound)]
        else:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]

        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / len(series)) * 100, 2)

        if outlier_count > 0:
            severity = "high" if outlier_pct > 10 else ("medium" if outlier_pct > 3 else "low")
            findings.append({
                "id": f"DQ-OUTLIER-{col}",
                "check": "outliers",
                "column": col,
                "severity": severity,
                "metric": {
                    "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2),
                    "min": float(series.min()),
                    "max": float(series.max()),
                },
                "evidence": f"Column '{col}' has {outlier_count} outliers ({outlier_pct}%) outside IQR bounds [{round(lower_bound, 2)}, {round(upper_bound, 2)}]. Extreme values: min={round(float(series.min()), 2)}, max={round(float(series.max()), 2)}.",
                "recommendation": "Inspect extreme values to verify if they represent valid domain tail events, measurement errors, or data corruption. Consider Winsorization or robust scaling.",
            })

    return findings



def detect_schema_anomalies(df: pd.DataFrame, schema: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    n_rows = len(df)

    for col, meta in schema.get("columns", {}).items():
        cardinality = meta.get("cardinality", 0)

        # Constant column
        if cardinality <= 1 and n_rows > 1:
            findings.append({
                "id": f"DQ-CONSTANT-{col}",
                "check": "constant_column",
                "column": col,
                "severity": "medium",
                "metric": {"cardinality": cardinality},
                "evidence": f"Column '{col}' has zero variance (only {cardinality} unique value).",
                "recommendation": "Drop this column prior to modeling as it carries zero variance or predictive information.",
            })

        # High cardinality categorical
        if meta.get("semantic_type") == "categorical" and cardinality > 50 and not meta.get("is_identifier"):
            findings.append({
                "id": f"DQ-HIGH-CARD-{col}",
                "check": "high_cardinality",
                "column": col,
                "severity": "medium",
                "metric": {"cardinality": cardinality},
                "evidence": f"Categorical column '{col}' has high cardinality ({cardinality} distinct categories).",
                "recommendation": "Group rare categories into an 'Other' bucket, or use target encoding rather than standard one-hot encoding.",
            })

    return findings



def calculate_data_quality_score(findings: list[dict[str, Any]], n_rows: int, n_cols: int) -> float:
    score = 100.0

    severity_penalties = {
        "critical": 15.0,
        "high": 8.0,
        "medium": 3.0,
        "low": 1.0,
        "info": 0.0,
    }

    for f in findings:
        sev = f.get("severity", "low")
        penalty = severity_penalties.get(sev, 1.0)
        score -= penalty

    return max(0.0, min(100.0, round(score, 1)))


