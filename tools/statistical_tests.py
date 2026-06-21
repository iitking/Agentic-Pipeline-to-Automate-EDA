
from typing import Any
import pandas as pd
import numpy as np
from scipy import stats


def compute_descriptive_stats(df: pd.DataFrame, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {"numerical": {}, "categorical": {}}

    # Numerical statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if schema and "type_groups" in schema:
        # Exclude pure identifiers
        id_cols = set(schema["type_groups"].get("identifier", []))
        numeric_cols = [c for c in numeric_cols if c not in id_cols]

    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue

        mean = float(series.mean())
        std = float(series.std()) if n > 1 else 0.0
        median = float(series.median())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        val_min = float(series.min())
        val_max = float(series.max())
        skewness = float(series.skew()) if n > 2 else 0.0
        kurt = float(series.kurtosis()) if n > 3 else 0.0
        cv = round((std / mean), 4) if mean != 0 else None

        results["numerical"][col] = {
            "count": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(median, 4),
            "min": round(val_min, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "max": round(val_max, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurt, 4),
            "coeff_variation": cv,
            "is_symmetric": abs(skewness) < 0.5,
            "skew_type": "right-skewed" if skewness > 0.5 else ("left-skewed" if skewness < -0.5 else "approximately symmetric"),
        }

    # Categorical distributions
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    if schema and "type_groups" in schema:
        cat_cols = [c for c in df.columns if c in schema["type_groups"].get("categorical", []) or c in schema["type_groups"].get("boolean", [])]

    for col in cat_cols:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue

        val_counts = series.value_counts()
        n_unique = int(series.nunique())
        top_cats = [
            {"category": str(k), "count": int(v), "percentage": round((v / n) * 100, 2)}
            for k, v in val_counts.head(10).items()
        ]

        results["categorical"][col] = {
            "count": n,
            "unique_count": n_unique,
            "top_categories": top_cats,
            "mode": str(val_counts.index[0]) if len(val_counts) > 0 else None,
            "mode_frequency_pct": round((val_counts.iloc[0] / n) * 100, 2) if len(val_counts) > 0 else 0,
        }

    return results



def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
    r, k = contingency_table.shape
    min_dim = min(r - 1, k - 1)
    if min_dim == 0:
        return 0.0
    v = np.sqrt(chi2 / (n * min_dim))
    return float(round(v, 4))



def compute_correlation_matrix(
    df: pd.DataFrame, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if schema and "type_groups" in schema:
        id_cols = set(schema["type_groups"].get("identifier", []))
        numeric_cols = [c for c in numeric_cols if c not in id_cols]

    # Clean numeric columns with at least 5 non-null rows
    valid_num_cols = [c for c in numeric_cols if df[c].dropna().nunique() > 1 and len(df[c].dropna()) >= 5]

    pearson_corr: dict[str, dict[str, float]] = {}
    spearman_corr: dict[str, dict[str, float]] = {}
    significant_pairs: list[dict[str, Any]] = []

    if len(valid_num_cols) >= 2:
        clean_num_df = df[valid_num_cols].dropna()
        if len(clean_num_df) >= 5:
            p_matrix = clean_num_df.corr(method="pearson").round(4)
            s_matrix = clean_num_df.corr(method="spearman").round(4)
            pearson_corr = p_matrix.to_dict()
            spearman_corr = s_matrix.to_dict()

            # Find top pairwise correlations (absolute > 0.3)
            for i in range(len(valid_num_cols)):
                for j in range(i + 1, len(valid_num_cols)):
                    c1 = valid_num_cols[i]
                    c2 = valid_num_cols[j]
                    val = p_matrix.loc[c1, c2]
                    if abs(val) >= 0.3:
                        significant_pairs.append({
                            "feature_a": c1,
                            "feature_b": c2,
                            "pearson_r": float(val),
                            "spearman_r": float(s_matrix.loc[c1, c2]),
                            "strength": "strong" if abs(val) >= 0.7 else ("moderate" if abs(val) >= 0.5 else "weak"),
                            "direction": "positive" if val > 0 else "negative",
                        })

    significant_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    # Categorical associations using Cramér's V
    cat_associations: list[dict[str, Any]] = []
    cat_cols = [
        c for c in df.columns
        if schema and c in schema["type_groups"].get("categorical", [])
        and 2 <= df[c].nunique() <= 20
    ] if schema else []

    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                ct = pd.crosstab(df[c1], df[c2])
                if ct.shape[0] > 1 and ct.shape[1] > 1:
                    cv = compute_cramers_v(ct)
                    if cv >= 0.2:
                        cat_associations.append({
                            "feature_a": c1,
                            "feature_b": c2,
                            "cramers_v": cv,
                            "strength": "strong" if cv >= 0.5 else ("moderate" if cv >= 0.3 else "weak"),
                        })

    return {
        "columns": valid_num_cols,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "top_numeric_correlations": significant_pairs[:15],
        "categorical_associations": cat_associations[:10],
    }



def test_normality(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    n = len(clean)
    if n < 8:
        return {"test": "none", "is_normal": False, "p_value": None, "note": "Sample size too small (n < 8)"}

    if n <= 5000:
        stat, p = stats.shapiro(clean)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(clean)
        test_name = "D'Agostino-Pearson"

    return {
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "is_normal": bool(p > 0.05),
        "sample_size": n,
    }



def compare_two_groups(
    group1: pd.Series, group2: pd.Series, group1_name: str, group2_name: str, feature_name: str
) -> dict[str, Any]:
    g1 = group1.dropna()
    g2 = group2.dropna()
    n1, n2 = len(g1), len(g2)

    if n1 < 5 or n2 < 5:
        return {
            "feature": feature_name,
            "test_type": "insufficient_sample",
            "p_value": None,
            "limitation": f"Sample sizes too small (n1={n1}, n2={n2}) for reliable testing.",
        }

    norm1 = test_normality(g1)["is_normal"]
    norm2 = test_normality(g2)["is_normal"]
    is_normal = norm1 and norm2

    mean1, mean2 = float(g1.mean()), float(g2.mean())
    median1, median2 = float(g1.median()), float(g2.median())

    if is_normal:
        # Check equal variance
        _, lew_p = stats.levene(g1, g2)
        equal_var = bool(lew_p > 0.05)
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
        test_name = "Two-Sample t-test" if equal_var else "Welch's t-test"

        # Cohen's d effect size
        s_pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / s_pooled if s_pooled > 0 else 0.0
        effect_label = "large" if abs(effect_size) > 0.8 else ("medium" if abs(effect_size) > 0.5 else "small")
    else:
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U test (non-parametric)"
        # Rank-biserial correlation
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        effect_label = "large" if abs(effect_size) > 0.5 else ("medium" if abs(effect_size) > 0.3 else "small")

    return {
        "feature": feature_name,
        "test_name": test_name,
        "group1": {"name": str(group1_name), "n": n1, "mean": round(mean1, 2), "median": round(median1, 2)},
        "group2": {"name": str(group2_name), "n": n2, "mean": round(mean2, 2), "median": round(median2, 2)},
        "p_value": round(float(p_val), 6),
        "is_significant": bool(p_val < 0.05),
        "effect_size": round(float(effect_size), 4),
        "effect_label": effect_label,
        "assumptions_checked": "Normality and variance equality tested",
        "caveat": "Statistical significance does not prove causal relationship; external confounding factors may exist.",
    }



def run_chi_square_test(series1: pd.Series, series2: pd.Series, name1: str, name2: str) -> dict[str, Any]:
    ct = pd.crosstab(series1, series2)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"test_name": "Chi-Square", "error": "Insufficient categories for 2D contingency table."}

    chi2, p_val, dof, expected = stats.chi2_contingency(ct)
    min_expected = float(np.min(expected))
    low_expected_warning = min_expected < 5.0
    cramers_v = compute_cramers_v(ct)

    return {
        "feature_a": name1,
        "feature_b": name2,
        "test_name": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_val), 6),
        "degrees_of_freedom": int(dof),
        "is_significant": bool(p_val < 0.05),
        "cramers_v": cramers_v,
        "min_expected_frequency": round(min_expected, 2),
        "assumption_warning": "Warning: Some expected frequencies < 5; results should be interpreted cautiously." if low_expected_warning else "Standard assumptions satisfied.",
    }


