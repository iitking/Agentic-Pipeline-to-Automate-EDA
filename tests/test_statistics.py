
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


