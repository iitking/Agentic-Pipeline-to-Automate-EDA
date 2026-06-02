"""Safe tool executor enforcing an allowlist of deterministic Python analysis tools."""

from typing import Any, Callable
import time
import pandas as pd
from tools.profiling import profile_dataset, detect_schema
from tools.data_quality import run_quality_audit
from tools.statistical_tests import (
    compute_descriptive_stats,
    compute_correlation_matrix,
    run_hypothesis_tests,
)
from tools.chart_generator import generate_eda_charts


class AllowlistedTools:
    """Registry of verified, safe analysis tools."""

    REGISTRY: dict[str, Callable[..., Any]] = {
        "profile_dataset": profile_dataset,
        "detect_schema": detect_schema,
        "run_quality_audit": run_quality_audit,
        "compute_descriptive_stats": compute_descriptive_stats,
        "compute_correlation_matrix": compute_correlation_matrix,
        "run_hypothesis_tests": run_hypothesis_tests,
        "generate_eda_charts": generate_eda_charts,
    }

    @classmethod
    def get_tool(cls, name: str) -> Callable[..., Any] | None:
        return cls.REGISTRY.get(name)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls.REGISTRY.keys())


class SafeExecutor:
    """Executes allowlisted tools with input validation, timeout protection, and structured responses."""

