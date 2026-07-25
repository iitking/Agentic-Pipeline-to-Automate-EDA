"""Insight Synthesis and Reporting Agent."""

from typing import Any
from workflow.state import EDAState
from workflow.prompts import INSIGHT_SYNTHESIS_SYSTEM_PROMPT, INSIGHT_SYNTHESIS_USER_PROMPT
from workflow.llm_client import LLMClient


class InsightAgent:
    """Agent that translates statistical outputs into evidence-backed business insights and recommendations."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _generate_heuristic_insights(self, state: EDAState) -> dict[str, Any]:
        """Generate deterministic, statistically grounded insights when LLM is unavailable."""
        profile = state.get("data_profile", {})
        quality_findings = state.get("quality_findings", [])
        quality_score = state.get("quality_score", 100.0)
        stats_data = state.get("statistical_results", {})
        hyp_tests = state.get("hypothesis_test_results", [])
        target = state.get("target_column")

        insights = []
        recommendations = []
        warnings = []

        # 1. Quality insight
        crit_issues = [f for f in quality_findings if f["severity"] in ["critical", "high"]]
        if crit_issues:
            issues_summary = "; ".join([i["evidence"] for i in crit_issues[:3]])
            insights.append({
                "category": "Data Quality",
                "headline": f"Data Quality Score: {quality_score}/100 with {len(crit_issues)} high-severity concerns",
                "observation": f"The dataset exhibits significant data hygiene challenges: {issues_summary}",
                "evidence": f"Total flagged quality issues: {len(quality_findings)}. Quality health score: {quality_score}%.",
                "caveats": "Unaddressed missingness and duplicate records can skew statistical inferences and induce bias.",
            })
            recommendations.append({
                "action_type": "Data Preprocessing",
                "recommendation": "Execute prioritized data deduplication and targeted imputation before downstream modeling.",
                "rationale": "High missingness or duplicate identifiers violate i.i.d. assumptions in most ML algorithms.",
            })
        else:
            insights.append({
                "category": "Data Quality",
                "headline": f"Strong Data Quality Health Score: {quality_score}/100",
                "observation": "The dataset exhibits low missingness and high structural consistency across records.",
                "evidence": f"Total rows: {profile.get('rows', 0)}, missing rate: {profile.get('missing_rate_pct', 0)}%.",
                "caveats": "Clean schema does not preclude subtle latent biases in sampling.",
            })

        # 2. Skewness / Distribution insight
        num_stats = stats_data.get("descriptive", {}).get("numerical", {})
        skewed_cols = [c for c, m in num_stats.items() if not m.get("is_symmetric", True)]
        if skewed_cols:
            skew_details = ", ".join([f"{c} (skew={num_stats[c]['skewness']})" for c in skewed_cols[:3]])
            insights.append({
                "category": "Distribution",
                "headline": f"Notable Asymmetry in Numerical Distributions: {len(skewed_cols)} skewed features",
