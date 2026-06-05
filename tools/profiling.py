
from pathlib import Path
from typing import Any
import re
import pandas as pd
import numpy as np


def load_dataset(file_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        # Try utf-8 first, fallback to latin-1
        try:
            df = pd.read_csv(path, nrows=max_rows)
        except UnicodeDecodeError:
            df = pd.read_csv(path, nrows=max_rows, encoding="latin-1")
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, nrows=max_rows)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats: .csv, .xlsx, .xls, .parquet"
        )

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return df



def detect_column_semantic_type(
    series: pd.Series, col_name: str, n_rows: int
) -> dict[str, Any]:
    clean_series = series.dropna()
    n_unique = clean_series.nunique()
    unique_ratio = n_unique / max(len(clean_series), 1)

    lower_col = str(col_name).lower()

    # 1. Check for identifier
    id_patterns = [r"id$", r"^id", r"_id$", r"identifier", r"customerid", r"userid", r"uuid", r"key$"]
    is_id_name = any(re.search(pat, lower_col) for pat in id_patterns)
    if (is_id_name and unique_ratio > 0.8) or (unique_ratio == 1.0 and n_rows > 30 and not pd.api.types.is_float_dtype(series)):
        return {
            "semantic_type": "identifier",
            "is_identifier": True,
            "confidence": 0.9 if is_id_name else 0.75,
            "cardinality": n_unique,
            "unique_ratio": float(unique_ratio),
        }

    # 2. Check for boolean / binary
    if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and set(clean_series.unique()).issubset({0, 1, "0", "1", True, False, "yes", "no", "Y", "N", "true", "false"})):
        return {
            "semantic_type": "boolean",
            "is_identifier": False,
            "confidence": 0.95,
            "cardinality": n_unique,
            "unique_ratio": float(unique_ratio),
        }

    # 3. Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "semantic_type": "datetime",
            "is_identifier": False,
            "confidence": 1.0,
            "cardinality": n_unique,
            "unique_ratio": float(unique_ratio),
        }

    # Check if object can be parsed as datetime
    if clean_series.dtype == object and ("date" in lower_col or "time" in lower_col or "year" in lower_col):
        try:
            pd.to_datetime(clean_series.head(50), errors="raise")
            return {
                "semantic_type": "datetime",
                "is_identifier": False,
                "confidence": 0.85,
                "cardinality": n_unique,
                "unique_ratio": float(unique_ratio),
            }
        except Exception:
            pass

    # 4. Check for numerical
    if pd.api.types.is_numeric_dtype(series):
        # Low cardinality integer may behave as categorical
        if n_unique <= 10 and n_rows >= 50 and pd.api.types.is_integer_dtype(series):
            return {
                "semantic_type": "categorical_numeric",
                "is_identifier": False,
                "confidence": 0.8,
                "cardinality": n_unique,
                "unique_ratio": float(unique_ratio),
            }
        return {
            "semantic_type": "numerical",
            "is_identifier": False,
            "confidence": 0.95,
            "cardinality": n_unique,
            "unique_ratio": float(unique_ratio),
        }

    # 5. Check for text vs categorical
    if clean_series.dtype == object or isinstance(clean_series.dtype, pd.CategoricalDtype):
        avg_str_len = clean_series.astype(str).map(len).mean() if len(clean_series) > 0 else 0
        if avg_str_len > 60 or (unique_ratio > 0.7 and n_unique > 50):
            return {
                "semantic_type": "text",
                "is_identifier": False,
                "confidence": 0.85,
                "cardinality": n_unique,
                "unique_ratio": float(unique_ratio),
            }
        return {
            "semantic_type": "categorical",
            "is_identifier": False,
            "confidence": 0.9,
            "cardinality": n_unique,
            "unique_ratio": float(unique_ratio),
        }

    return {
        "semantic_type": "unknown",
        "is_identifier": False,
        "confidence": 0.5,
        "cardinality": n_unique,
        "unique_ratio": float(unique_ratio),
    }



def infer_potential_target(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    candidate_keywords = [
        "target", "label", "class", "churn", "exited", "status", "outcome",
        "survived", "default", "attrition", "converted", "is_churn", "response",
        "sale_price", "price", "revenue", "loss", "rating", "fraud"
    ]

    candidates = []
    for col, meta in schema["columns"].items():
        if meta.get("is_identifier"):
            continue

        lower_col = col.lower()
        score = 0.0

        # Exact or partial match with common target keywords
        for kw in candidate_keywords:
            if lower_col == kw:
                score += 0.85
                break
            elif kw in lower_col:
                score += 0.5
                break

        # Suffix matching
        if lower_col.endswith("_target") or lower_col.endswith("_label") or lower_col.endswith("_status"):
            score += 0.7

        # Placement heuristic: Often the last column
        if col == df.columns[-1]:
            score += 0.2

        # Binary/categorical targets are very common
        sem_type = meta["semantic_type"]
        if sem_type in ["boolean", "categorical_numeric"] and meta["cardinality"] == 2:
            score += 0.2
        elif sem_type == "categorical" and 2 <= meta["cardinality"] <= 10:
            score += 0.15

        if score >= 0.5:
            target_type = "classification" if sem_type in ["boolean", "categorical", "categorical_numeric"] or meta["cardinality"] <= 15 else "regression"
            candidates.append({
                "column": col,
                "score": round(score, 2),
                "type": target_type,
                "semantic_type": sem_type,
            })

    if not candidates:
        return None

    # Sort descending by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    return {
        "column": best["column"],
        "target_type": best["type"],
        "confidence": best["score"],
        "all_candidates": candidates,
    }



def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    columns_info: dict[str, Any] = {}

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    identifier_cols: list[str] = []
    text_cols: list[str] = []
    boolean_cols: list[str] = []

    for col in df.columns:
        s = df[col]
        semantic = detect_column_semantic_type(s, col, n_rows)
        dtype_str = str(s.dtype)
        null_count = int(s.isnull().sum())
        null_pct = float(round((null_count / max(n_rows, 1)) * 100, 2))
        n_unique = int(s.nunique(dropna=True))

        col_meta = {
            "dtype": dtype_str,
            "semantic_type": semantic["semantic_type"],
            "is_identifier": semantic["is_identifier"],
            "cardinality": n_unique,
            "unique_ratio": round(semantic["unique_ratio"], 4),
            "null_count": null_count,
            "null_pct": null_pct,
            "memory_bytes": int(s.memory_usage(deep=True)),
        }
        columns_info[col] = col_meta

        stype = semantic["semantic_type"]
        if semantic["is_identifier"]:
            identifier_cols.append(col)
        elif stype in ["numerical", "categorical_numeric"]:
            numerical_cols.append(col)
        elif stype == "categorical":
            categorical_cols.append(col)
        elif stype == "datetime":
            datetime_cols.append(col)
        elif stype == "boolean":
            boolean_cols.append(col)
        elif stype == "text":
            text_cols.append(col)

    schema = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "columns": columns_info,
        "type_groups": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "identifier": identifier_cols,
            "text": text_cols,
            "boolean": boolean_cols,
        },
    }

    inferred_target = infer_potential_target(df, schema)
    schema["inferred_target"] = inferred_target
    return schema


