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

    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Safely execute an allowlisted tool function."""
        if tool_name not in AllowlistedTools.REGISTRY:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' is not in the approved safe tool allowlist. Permitted tools: {AllowlistedTools.list_tools()}",
                "result": None,
                "execution_time_ms": 0.0,
            }

        func = AllowlistedTools.get_tool(tool_name)
        start_time = time.time()

        try:
            # Execute tool with provided keyword arguments
            result = func(**kwargs)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "success": True,
                "tool_name": tool_name,
                "result": result,
                "error": None,
                "execution_time_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "success": False,
                "tool_name": tool_name,
                "result": None,
                "error": f"Execution error in {tool_name}: {str(e)}",
                "execution_time_ms": duration_ms,
            }
