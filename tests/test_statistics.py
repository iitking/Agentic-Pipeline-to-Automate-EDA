
import pytest
import pandas as pd
import numpy as np
from tools.statistical_tests import (
    compute_descriptive_stats,
    compute_correlation_matrix,
    compare_two_groups,
    compute_cramers_v,
    run_hypothesis_tests,
)


def test_compute_descriptive_stats():
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["A", "B", "A", "A", "C"],
    })
    res = compute_descriptive_stats(df)
    assert "numerical" in res
    assert "categorical" in res

    num_stats = res["numerical"]["num"]
    assert num_stats["mean"] == 30.0
    assert num_stats["median"] == 30.0
    assert num_stats["min"] == 10.0
    assert num_stats["max"] == 50.0

    cat_stats = res["categorical"]["cat"]
    assert cat_stats["mode"] == "A"
    assert cat_stats["unique_count"] == 3



def test_compute_correlation_matrix():
    x = np.linspace(1, 100, 50)
    y = 2 * x + np.random.normal(0, 2, 50)
    df = pd.DataFrame({"x": x, "y": y})
    res = compute_correlation_matrix(df)

    assert "pearson" in res
    assert "spearman" in res
    r = res["pearson"]["x"]["y"]
    assert r > 0.9  # Strong positive correlation



def test_compare_two_groups_welch_or_mannwhitney():
    # Group 1 with low values, Group 2 with high values
    g1 = pd.Series([10, 11, 12, 10, 9, 11, 10, 12, 11, 10])
    g2 = pd.Series([50, 52, 49, 51, 50, 53, 50, 51, 52, 50])

    res = compare_two_groups(g1, g2, "G1", "G2", "feature_test")
    assert res["is_significant"] is True
    assert res["p_value"] < 0.001
    assert res["effect_label"] == "large"


