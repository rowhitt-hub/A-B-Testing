import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.proportion as smprop
import statsmodels.stats.power as smp


def z_test_proportions(success_a, n_a, success_b, n_b, alpha=0.05, alternative='two-sided'):
    """
    Two-sample z-test for proportions (binary conversion metric).
    Returns p-value, confidence interval, Cohen's h, and group rates.
    """
    if n_a == 0 or n_b == 0:
        return {
            'z_stat': np.nan, 'p_value': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'diff': np.nan, 'p_a': np.nan, 'p_b': np.nan,
            'effect_size': np.nan, 'success_a': success_a, 'success_b': success_b,
            'sample_a': n_a, 'sample_b': n_b
        }

    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)

    # Pooled standard error under null
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se_pool if se_pool > 0 else 0.0

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        z_crit = stats.norm.ppf(1 - alpha / 2)
    elif alternative == 'larger':
        p_value = 1 - stats.norm.cdf(z)
        z_crit = stats.norm.ppf(1 - alpha)
    elif alternative == 'smaller':
        p_value = stats.norm.cdf(z)
        z_crit = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")

    # Unpooled standard error for CI
    se_unpool = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_a - p_b
    ci_lower = diff - z_crit * se_unpool
    ci_upper = diff + z_crit * se_unpool

    # Cohen's h effect size
    effect_size = 2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b))

    return {
        'z_stat': z,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'diff': diff,
        'p_a': p_a,
        'p_b': p_b,
        'effect_size': effect_size,
        'success_a': success_a,
        'success_b': success_b,
        'sample_a': n_a,
        'sample_b': n_b
    }


def t_test_ind(group_a, group_b, alpha=0.05):
    """
    Welch's t-test for continuous metrics (e.g., total ads).
    Returns p-value, confidence interval, and Cohen's d.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Welch-Satterthwaite degrees of freedom
    se = np.sqrt(var_a / n_a + var_b / n_b)
    df = ((var_a / n_a + var_b / n_b) ** 2) / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    ) if n_a > 1 and n_b > 1 else n_a + n_b - 2

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean_a - mean_b
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    # Cohen's d
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohen_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'diff': diff,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'effect_size': cohen_d
    }


def chi_square_test(table):
    """
    Chi-square test of independence for 2x2 or larger contingency tables.
    """
    table = np.asarray(table)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    min_dim = min(table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    return {
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'expected': expected,
        'cramer_v': cramer_v
    }


def power_analysis_proportions(p1, p2, alpha=0.05, power=0.8, alternative='two-sided'):
    """
    Required sample size per group for a two-sample proportion test.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return np.nan, np.nan

    effect_size = smprop.proportion_effectsize(p1, p2)
    nobs = smp.NormalIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1,
        alternative=alternative
    )
    return math.ceil(nobs), effect_size


def simulate_peeking_inflation(n_per_group=1000, p=0.05, n_peeks=5,
                               n_simulations=1000, seed=42):
    """
    Simulates the false positive rate when peeking at data multiple times
    and stopping if p < 0.05 under the null hypothesis.
    """
    rng = np.random.default_rng(seed)
    false_positives = 0
    peek_sizes = np.linspace(0, n_per_group, n_peeks + 1, dtype=int)[1:]

    for _ in range(n_simulations):
        group_a = rng.binomial(1, p, n_per_group)
        group_b = rng.binomial(1, p, n_per_group)

        for size in peek_sizes:
            if size == 0:
                continue
            res = z_test_proportions(
                group_a[:size].sum(), size,
                group_b[:size].sum(), size
            )
            if res['p_value'] < 0.05:
                false_positives += 1
                break

    return false_positives / n_simulations


def decision_from_test(p_value, ci_lower, ci_upper, effect_size,
                       alpha=0.05, practical_lift=0.01):
    """
    Translates statistical output into a business decision.
    """
    if np.isnan(p_value):
        return "Invalid test — check sample sizes and data."

    if p_value >= alpha:
        return "No significant difference — do not ship / need more data."

    if ci_lower > practical_lift:
        return "Ship: statistically significant and practically meaningful positive lift."

    if ci_upper < -practical_lift:
        return "Don't ship: statistically significant but negative impact."

    return "Statistically significant but effect may not be practically meaningful — need more data/context."


def build_results_table(control_label, treat_label, n_a, n_b, x_a, x_b,
                        p_value, ci_lower, ci_upper, effect_size, decision):
    """
    Builds a results DataFrame with a 'result' column describing the quantitative outcome.
    """
    rows = [
        {
            'group': control_label,
            'sample_size': n_a,
            'conversions': x_a,
            'conversion_rate': x_a / n_a if n_a else np.nan,
            'result': 'control'
        },
        {
            'group': treat_label,
            'sample_size': n_b,
            'conversions': x_b,
            'conversion_rate': x_b / n_b if n_b else np.nan,
            'result': 'treatment'
        },
        {
            'group': 'Difference (treat - control)',
            'sample_size': n_a + n_b,
            'conversions': x_a + x_b,
            'conversion_rate': (x_b / n_b - x_a / n_a) if n_a and n_b else np.nan,
            'result': decision
        }
    ]
    return pd.DataFrame(rows)


def build_continuous_results_table(control_label, treat_label, n_a, n_b,
                                   mean_a, mean_b, p_value, ci_lower, ci_upper,
                                   effect_size, decision):
    """
    Builds a results DataFrame for continuous metrics with a 'result' column.
    """
    rows = [
        {
            'group': control_label,
            'sample_size': n_a,
            'mean': mean_a,
            'result': 'control'
        },
        {
            'group': treat_label,
            'sample_size': n_b,
            'mean': mean_b,
            'result': 'treatment'
        },
        {
            'group': 'Difference (treat - control)',
            'sample_size': n_a + n_b,
            'mean': mean_b - mean_a,
            'result': decision
        }
    ]
    return pd.DataFrame(rows)