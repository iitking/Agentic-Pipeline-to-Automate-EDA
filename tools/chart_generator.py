
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



def plot_distributions(
    df: pd.DataFrame, output_dir: Path, schema: dict[str, Any] | None = None, max_plots: int = 4
) -> list[dict[str, Any]]:
    charts = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if schema and "type_groups" in schema:
        id_cols = set(schema["type_groups"].get("identifier", []))
        num_cols = [c for c in num_cols if c not in id_cols]

    for col in num_cols[:max_plots]:
        series = df[col].dropna()
        if series.nunique() <= 2:
            continue

        fig = px.histogram(
            df,
            x=col,
            marginal="box",
            nbins=30,
            title=f"Distribution of {col}",
            template="plotly_white",
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)

        file_name = f"dist_{col.lower().replace(' ', '_')}.html"
        file_path = output_dir / file_name
        fig.write_html(str(file_path), include_plotlyjs="cdn", full_html=False)

        charts.append({
            "chart_type": "distribution_histogram",
            "title": f"Distribution: {col}",
            "file_path": str(file_path),
            "html_content": fig.to_html(include_plotlyjs=False, full_html=False),
            "columns": [col],
            "description": f"Histogram and boxplot showing spread, skew, and potential outliers in {col}.",
        })

    return charts



def plot_categorical_breakdowns(
    df: pd.DataFrame, output_dir: Path, schema: dict[str, Any] | None = None, max_plots: int = 3
) -> list[dict[str, Any]]:
    charts = []
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if schema and "type_groups" in schema:
        id_cols = set(schema["type_groups"].get("identifier", []))
        cat_cols = [c for c in cat_cols if c not in id_cols]

    for col in cat_cols[:max_plots]:
        if df[col].nunique() > 25:
            continue

        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]

        fig = px.bar(
            counts.head(10),
            x=col,
            y="Count",
            title=f"Category Counts: {col}",
            template="plotly_white",
            color="Count",
            color_continuous_scale="Blues",
            text="Count",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
        fig.update_traces(textposition="outside")

        file_name = f"cat_{col.lower().replace(' ', '_')}.html"
        file_path = output_dir / file_name
        fig.write_html(str(file_path), include_plotlyjs="cdn", full_html=False)

        charts.append({
            "chart_type": "categorical_bar",
            "title": f"Categorical Count: {col}",
            "file_path": str(file_path),
            "html_content": fig.to_html(include_plotlyjs=False, full_html=False),
            "columns": [col],
            "description": f"Frequency distribution across distinct levels of {col}.",
        })

    return charts


