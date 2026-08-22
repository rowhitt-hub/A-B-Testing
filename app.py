import streamlit as st
import pandas as pd
import numpy as np

from data_gen import load_marketing_ab_data
from stats_engine import (
    z_test_proportions,
    t_test_ind,
    power_analysis_proportions,
    simulate_peeking_inflation,
    decision_from_test,
    build_results_table,
    build_continuous_results_table
)


st.set_page_config(page_title="Marketing A/B Test Dashboard", layout="wide")
st.title("Marketing A/B Test Dashboard")
st.markdown("Data: Kaggle **Marketing A/B Testing** dataset (ad vs psa).")


@st.cache_data
def get_data():
    return load_marketing_ab_data()


df = get_data()

st.subheader("Raw Data Preview")
st.dataframe(df.head())


# ---------------------------------------------------------------------
# Sidebar: Experiment design / sample size calculator
# ---------------------------------------------------------------------
st.sidebar.header("Experiment Design: Sample Size Calculator")

baseline = st.sidebar.number_input(
    "Baseline conversion rate",
    min_value=0.0001, max_value=0.999, value=0.025,
    step=0.001, format="%.4f"
)
lift = st.sidebar.number_input(
    "Minimum detectable lift (absolute)",
    min_value=0.0001, max_value=0.5, value=0.01,
    step=0.001, format="%.4f"
)
alpha = st.sidebar.number_input(
    "Alpha (Type I error)",
    min_value=0.01, max_value=0.20, value=0.05, step=0.01
)
power = st.sidebar.number_input(
    "Power (1 - Type II)",
    min_value=0.50, max_value=0.99, value=0.80, step=0.01
)

p2 = min(baseline + lift, 0.999)
required_n, effect_size_ss = power_analysis_proportions(baseline, p2, alpha, power)
st.sidebar.write(f"Required sample per group: **{required_n:,}**")
st.sidebar.write(f"Cohen's h: {effect_size_ss:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sequential Testing / Peeking Problem**")

n_peeks = st.sidebar.slider("Number of interim peeks", 2, 20, 5)
if st.sidebar.button("Simulate false positive inflation"):
    with st.spinner("Simulating peeking..."):
        fpr = simulate_peeking_inflation(
            n_per_group=1000, p=0.05, n_peeks=n_peeks,
            n_simulations=1000, seed=42
        )
    st.sidebar.write(f"Observed false positive rate: **{fpr:.3f}** (nominal 0.05)")
    st.sidebar.warning(
        "Peeking at the data and stopping when p < 0.05 inflates Type I error. "
        "Use pre-registered stopping rules or alpha spending."
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
st.subheader("A/B Test Analysis")

groups = sorted(df['test_group'].unique().tolist())
col1, col2 = st.columns(2)

with col1:
    control_group = st.selectbox(
        "Control group",
        groups,
        index=groups.index('psa') if 'psa' in groups else 0
    )
with col2:
    treatment_group = st.selectbox(
        "Treatment group",
        groups,
        index=groups.index('ad') if 'ad' in groups else 0
    )

metric = st.radio("Metric", ["Conversion rate (binary)", "Total ads (continuous)"])

if control_group == treatment_group:
    st.error("Please choose different groups for control and treatment.")
else:
    ctrl_mask = df['test_group'] == control_group
    treat_mask = df['test_group'] == treatment_group

    if metric == "Conversion rate (binary)":
        x_ctrl = int(df.loc[ctrl_mask, 'converted'].sum())
        n_ctrl = int(ctrl_mask.sum())
        x_treat = int(df.loc[treat_mask, 'converted'].sum())
        n_treat = int(treat_mask.sum())

        res = z_test_proportions(x_ctrl, n_ctrl, x_treat, n_treat, alpha=alpha)

        st.write(f"Control ({control_group}): {x_ctrl}/{n_ctrl} = {res['p_a']:.4f}")
        st.write(f"Treatment ({treatment_group}): {x_treat}/{n_treat} = {res['p_b']:.4f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("p-value", f"{res['p_value']:.4f}")
        c2.metric("Cohen's h", f"{res['effect_size']:.4f}")
        c3.metric("95% CI for difference", f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")

        decision = decision_from_test(
            res['p_value'], res['ci_lower'], res['ci_upper'],
            res['effect_size'], alpha=alpha, practical_lift=lift
        )

        st.subheader("Business Decision")
        st.write(decision)

        if min(n_ctrl, n_treat) < required_n:
            st.warning(
                f"Sample size is below the required {required_n:,} per group. "
                "Consider 'need more data' before making a final decision."
            )

        results_df = build_results_table(
            control_group, treatment_group,
            n_ctrl, n_treat, x_ctrl, x_treat,
            res['p_value'], res['ci_lower'], res['ci_upper'],
            res['effect_size'], decision
        )
        st.subheader("Results Table")
        st.dataframe(results_df)

    else:  # Total ads (continuous)
        a_metric = df.loc[ctrl_mask, 'total_ads'].astype(float)
        b_metric = df.loc[treat_mask, 'total_ads'].astype(float)

        res = t_test_ind(a_metric, b_metric, alpha=alpha)

        st.write(f"Control mean: {res['mean_a']:.2f}")
        st.write(f"Treatment mean: {res['mean_b']:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("p-value", f"{res['p_value']:.4f}")
        c2.metric("Cohen's d", f"{res['effect_size']:.4f}")
        c3.metric("95% CI for mean difference", f"[{res['ci_lower']:.2f}, {res['ci_upper']:.2f}]")

        decision = decision_from_test(
            res['p_value'], res['ci_lower'], res['ci_upper'],
            res['effect_size'], alpha=alpha, practical_lift=0.1
        )

        st.subheader("Business Decision")
        st.write(decision)

        results_df = build_continuous_results_table(
            control_group, treatment_group,
            len(a_metric), len(b_metric),
            res['mean_a'], res['mean_b'],
            res['p_value'], res['ci_lower'], res['ci_upper'],
            res['effect_size'], decision
        )
        st.subheader("Results Table")
        st.dataframe(results_df)