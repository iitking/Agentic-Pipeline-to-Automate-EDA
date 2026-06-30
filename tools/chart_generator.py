
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def ensure_output_dir(dir_path: str | Path) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if null_cols.empty:
        return None

    pcts = (null_cols / len(df) * 100).round(2).reset_index()
    pcts.columns = ["Column", "Missing_Percentage"]
    pcts = pcts.sort_values(by="Missing_Percentage", ascending=True)

    fig = px.bar(
        pcts,
        x="Missing_Percentage",
        y="Column",
        orientation="h",
        title="Missing Values by Feature (%)",
        text="Missing_Percentage",
        color="Missing_Percentage",
        color_continuous_scale="Reds",
        labels={"Missing_Percentage": "Missing (%)"},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=350)
    fig.update_traces(texttemplate="%{text}%", textposition="outside")

    file_path = output_dir / "missing_values.html"
    fig.write_html(str(file_path), include_plotlyjs="cdn", full_html=False)

    return {
        "chart_type": "missing_values_bar",
        "title": "Missing Values Overview",
        "file_path": str(file_path),
        "html_content": fig.to_html(include_plotlyjs=False, full_html=False),
        "columns": pcts["Column"].tolist(),
        "description": f"Missing value distribution across {len(null_cols)} affected columns.",
    }



def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path, schema: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if schema and "type_groups" in schema:
        id_cols = set(schema["type_groups"].get("identifier", []))
        num_cols = [c for c in num_cols if c not in id_cols]

    valid_cols = [c for c in num_cols if df[c].dropna().nunique() > 1]
    if len(valid_cols) < 2:
        return None

    corr_df = df[valid_cols].dropna().corr().round(2)
    if corr_df.empty:
        return None

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix (Pearson)",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=450)

    file_path = output_dir / "correlation_heatmap.html"
    fig.write_html(str(file_path), include_plotlyjs="cdn", full_html=False)

    return {
        "chart_type": "correlation_heatmap",
        "title": "Correlation Heatmap",
        "file_path": str(file_path),
        "html_content": fig.to_html(include_plotlyjs=False, full_html=False),
        "columns": valid_cols,
        "description": "Pairwise Pearson correlation coefficients between numeric variables.",
    }


