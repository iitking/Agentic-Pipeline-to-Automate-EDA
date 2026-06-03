
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


