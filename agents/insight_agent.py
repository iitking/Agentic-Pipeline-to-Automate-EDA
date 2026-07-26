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
                "observation": f"The following features deviate significantly from a normal distribution: {skew_details}.",
                "evidence": f"Evaluated using Fisher-Pearson skewness coefficient (|skew| > 0.5).",
                "caveats": "Linear models assume normality in residuals, not necessarily raw features, but extreme skew can inflate leverage.",
            })
            recommendations.append({
                "action_type": "Feature Engineering",
                "recommendation": f"Apply log1p or Yeo-Johnson power transformation to skewed features ({', '.join(skewed_cols[:3])}).",
                "rationale": "Stabilizes variance and reduces the impact of heavy tails on distance-based and linear estimators.",
            })

        # 3. Correlation insight
        corrs = stats_data.get("correlations", {}).get("top_numeric_correlations", [])
        if corrs:
            top_corr = corrs[0]
            insights.append({
                "category": "Correlation",
                "headline": f"Strongest Feature Pairwise Association: {top_corr['feature_a']} and {top_corr['feature_b']}",
                "observation": f"{top_corr['feature_a']} and {top_corr['feature_b']} exhibit a {top_corr['strength']} {top_corr['direction']} correlation (Pearson r = {top_corr['pearson_r']}).",
                "evidence": f"Spearman rank correlation r_s = {top_corr['spearman_r']}.",
                "caveats": "Correlation indicates linear or monotonic co-movement, not causal dependence.",
            })
            if abs(top_corr["pearson_r"]) > 0.8:
                recommendations.append({
                    "action_type": "Feature Engineering",
                    "recommendation": f"Inspect potential multicollinearity between {top_corr['feature_a']} and {top_corr['feature_b']}.",
                    "rationale": "High collinearity (>0.8) inflates parameter variance in regression models.",
                })

        # 4. Target relationship insight
        sig_tests = [t for t in hyp_tests if t.get("is_significant")]
        if sig_tests and target:
            t0 = sig_tests[0]
            feat = t0.get("feature") or t0.get("feature_a")
            p_val = t0.get("p_value")
            eff = t0.get("effect_label", "moderate")
            insights.append({
                "category": "Target Relationship",
                "headline": f"Significant Divergence in {feat} Across {target} Classes",
                "observation": f"Feature '{feat}' demonstrated statistically significant variation between '{target}' segments (p-value = {p_val}, {eff} effect size).",
                "evidence": f"Evaluated using {t0.get('test_name')}. Group 1 mean: {t0.get('group1', {}).get('mean')}, Group 2 mean: {t0.get('group2', {}).get('mean')}.",
                "caveats": "Statistical significance does not prove that changing this feature will directly alter the target outcome.",
            })
            recommendations.append({
                "action_type": "Modeling Strategy",
                "recommendation": f"Include '{feat}' as a primary candidate feature in baseline predictive models for '{target}'.",
                "rationale": f"Demonstrated empirical discriminatory power with p < 0.05.",
            })

        exec_summary = (
            f"Dataset '{profile.get('dataset_name', '')}' contains {profile.get('rows', 0)} records across "
