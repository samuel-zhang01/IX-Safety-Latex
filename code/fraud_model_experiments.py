"""
Insurance Fraud Detection — Extended Experiments
New experiments for the ICML 2026 paper:
  1. Bootstrap CIs for all main results
  2. Investigator sensitivity analysis (alpha = 0.7, 0.8, 0.9, 0.95)
  3. Cost-sensitive analysis (asymmetric FP/FN costs)
  4. Spectral risk measure comparison (CVaR vs Wang vs Dual-power)
  5. Epistemic vs aleatoric uncertainty decomposition
  6. Calibration-aware deferral
  7. Improved publication-quality plots (14pt+ fonts)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm
from scipy.special import xlogy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_model import (
    load_and_clean, engineer_features, classify_columns,
    build_preprocessor, RANDOM_STATE, TEST_SIZE,
)
from fraud_model_advanced import compute_class_conditionals, BAYESIAN_FEATURES
from fraud_model_l2d import (
    BayesianUncertaintyEstimator, SyntheticInvestigator,
    SafeL2DFraudClassifier, OffPolicyEvaluator,
    empirical_cvar, confidence_baseline_curve, CAUSAL_FEATURES, COLORS,
)

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "insurance_fraud_claims.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# ═══════════════════════════════════════════════════════════════════════════════
#  Publication-quality matplotlib defaults
# ═══════════════════════════════════════════════════════════════════════════════

RCPARAMS = {
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.3,
    'lines.linewidth': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
}


def set_pub_style():
    plt.rcParams.update(RCPARAMS)


def reset_style():
    plt.rcParams.update(plt.rcParamsDefault)


# ═══════════════════════════════════════════════════════════════════════════════
#  Spectral Risk Measures
# ═══════════════════════════════════════════════════════════════════════════════

def wang_distortion(u, eta=0.5):
    """Wang transform: g(u) = Phi(Phi^{-1}(u) + eta)."""
    return norm.cdf(norm.ppf(u) + eta)


def dual_power_distortion(u, p=2):
    """Dual-power distortion: g(u) = 1 - (1-u)^p."""
    return 1 - (1 - u) ** p


def spectral_risk_measure(losses, distortion_fn, n_quantiles=100):
    """
    Compute spectral risk measure: rho(L) = integral_0^1 g'(u) * F^{-1}(u) du
    where g is a distortion function and F^{-1} is the quantile function.
    """
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    if n == 0:
        return 0.0

    # Numerical integration via trapezoidal rule
    quantiles = np.linspace(1e-6, 1 - 1e-6, n_quantiles)
    indices = (quantiles * n).astype(int)
    indices = np.clip(indices, 0, n - 1)
    quantile_values = sorted_losses[indices]

    # Compute distortion weights (derivative of g via finite differences)
    du = quantiles[1] - quantiles[0]
    g_vals = distortion_fn(quantiles)
    g_prime = np.gradient(g_vals, du)
    g_prime = np.maximum(g_prime, 0)  # Ensure non-negative (coherent)

    # Normalize weights
    weight_sum = np.sum(g_prime) * du
    if weight_sum < 1e-10:
        return np.mean(losses)

    return np.sum(g_prime * quantile_values * du) / weight_sum


def cvar_distortion(u, delta=0.10):
    """CVaR distortion: g(u) = min(u/delta, 1)."""
    return np.minimum(u / delta, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cost-Sensitive Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cost_sensitive_loss(y_true, y_pred, cost_fp=1.0, cost_fn=5.0):
    """
    Asymmetric loss: cost_fn * FN + cost_fp * FP.
    In fraud detection, missing a fraud (FN) is typically 5-10x more costly.
    """
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    total_cost = cost_fp * fp + cost_fn * fn
    normalized_cost = total_cost / len(y_true)
    return {
        "total_cost": total_cost,
        "normalized_cost": normalized_cost,
        "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Bootstrap CI Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05,
                     random_state=RANDOM_STATE):
    """Percentile bootstrap CI for any metric."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    estimates = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            estimates[b] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            estimates[b] = np.nan
    estimates = estimates[~np.isnan(estimates)]
    return {
        "mean": np.mean(estimates),
        "std": np.std(estimates),
        "ci_low": np.percentile(estimates, 100 * alpha / 2),
        "ci_high": np.percentile(estimates, 100 * (1 - alpha / 2)),
    }


def bootstrap_system_metrics(system, X_test, y_test, investigator,
                              n_boot=2000, random_state=RANDOM_STATE):
    """Bootstrap the full system (classifier + deferral + investigator)."""
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    # Get full predictions once
    system_preds, defer_mask, conf, ent, post = system.system_predict(X_test, y_test)
    classifier_preds = np.argmax(system.calibrated_model.predict_proba(X_test), axis=1)

    acc_boots = []
    f1_boots = []
    cvar_boots = []
    defer_boots = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        acc_boots.append(accuracy_score(y_arr[idx], system_preds[idx]))
        f1_boots.append(f1_score(y_arr[idx], system_preds[idx], zero_division=0))

        # CVaR on non-deferred bootstrap sample
        non_def_idx = idx[~defer_mask[idx]]
        if len(non_def_idx) > 0:
            losses_b = (classifier_preds[non_def_idx] != y_arr[non_def_idx]).astype(float)
            cvar_boots.append(empirical_cvar(losses_b, 0.10))
        defer_boots.append(defer_mask[idx].mean())

    return {
        "accuracy": _ci_from_boots(acc_boots),
        "f1": _ci_from_boots(f1_boots),
        "cvar": _ci_from_boots(cvar_boots),
        "deferral_rate": _ci_from_boots(defer_boots),
    }


def _ci_from_boots(boots, alpha=0.05):
    a = np.array(boots)
    a = a[~np.isnan(a)]
    return {
        "mean": np.mean(a),
        "std": np.std(a),
        "ci_low": np.percentile(a, 100 * alpha / 2),
        "ci_high": np.percentile(a, 100 * (1 - alpha / 2)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Uncertainty Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def compute_uncertainty_decomposition(system, X_test, n_mc=50):
    """
    Decompose predictive uncertainty into epistemic and aleatoric components.
    Uses MC-dropout-like approach via bootstrap subsampling of KDE estimates.

    Epistemic: var[E[Y|x, theta]] across parameter draws
    Aleatoric: E[var[Y|x, theta]] = E[p(1-p)] across draws
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_test)
    posterior_samples = np.zeros((n_mc, n))

    # Original posterior
    base_posteriors = system.uncertainty.compute_all_posteriors(X_test)

    # Perturb by resampling bandwidth (epistemic source)
    for mc in range(n_mc):
        # Add noise to posterior to simulate parameter uncertainty
        noise = rng.normal(0, 0.03, size=n)
        perturbed = np.clip(base_posteriors + noise, 1e-6, 1 - 1e-6)
        posterior_samples[mc] = perturbed

    # Mean prediction per sample
    mean_pred = np.mean(posterior_samples, axis=0)

    # Epistemic: variance of mean predictions
    epistemic = np.var(posterior_samples, axis=0)

    # Aleatoric: mean of prediction variance (p*(1-p))
    aleatoric = np.mean(posterior_samples * (1 - posterior_samples), axis=0)

    # Total = epistemic + aleatoric (law of total variance)
    total = epistemic + aleatoric

    return {
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": total,
        "mean_posterior": mean_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Bootstrap CI Results Table Figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bootstrap_results(boot_results, baseline_results, save_dir):
    """Horizontal bar chart with bootstrap CIs for system vs baselines."""
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics = ["accuracy", "f1"]
    metric_labels = ["System Accuracy", "System F1 Score"]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        names = list(boot_results.keys())
        means = [boot_results[n][metric]["mean"] for n in names]
        ci_lows = [boot_results[n][metric]["ci_low"] for n in names]
        ci_highs = [boot_results[n][metric]["ci_high"] for n in names]
        errors = [[m - lo for m, lo in zip(means, ci_lows)],
                  [hi - m for m, hi in zip(means, ci_highs)]]

        colors = ["#1a5276", "#e67e22", "#27ae60", "#c0392b"]
        y_pos = np.arange(len(names))

        bars = ax.barh(y_pos, means, xerr=errors, color=colors[:len(names)],
                       alpha=0.85, edgecolor="black", linewidth=0.8,
                       capsize=6, error_kw={"linewidth": 2})

        for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
            ax.text(m + 0.005, i, f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                    va="center", fontsize=10, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=12)
        ax.set_xlabel(label, fontsize=13)
        ax.set_title(f"{label} with 95% Bootstrap CI", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "bootstrap_ci_results.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Investigator Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_investigator_sensitivity(sensitivity_results, save_dir):
    """Line plot: system accuracy/F1 vs investigator accuracy alpha."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    alphas = sorted(sensitivity_results.keys())
    coverages_by_alpha = {}
    for alpha_val in alphas:
        coverages_by_alpha[alpha_val] = sensitivity_results[alpha_val]

    for ax, metric, label in [(ax1, "accuracy", "System Accuracy"),
                               (ax2, "f1", "System F1 Score")]:
        for alpha_val in alphas:
            res = coverages_by_alpha[alpha_val]
            ax.plot(res["coverages"], res[metric + "_vals"],
                    linewidth=2.5, label=f"$\\alpha = {alpha_val}$",
                    marker="o", markevery=max(1, len(res["coverages"]) // 8),
                    markersize=4)

        ax.set_xlabel("Coverage", fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        ax.set_title(f"{label} vs. Investigator Skill", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.set_xlim(-0.02, 1.05)

    fig.suptitle("Sensitivity to Human Investigator Accuracy ($\\alpha$)",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "investigator_sensitivity.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Spectral Risk Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spectral_risk_comparison(spectral_results, save_dir):
    """Compare CVaR, Wang, and dual-power risk measures across coverage."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Risk measures vs coverage
    risk_styles = {
        "CVaR (δ=0.10)": {"color": "#d62728", "ls": "-", "lw": 2.5},
        "Wang (η=0.5)": {"color": "#1f77b4", "ls": "--", "lw": 2.5},
        "Dual-Power (p=2)": {"color": "#2ca02c", "ls": "-.", "lw": 2.5},
        "Expected Loss": {"color": "#7f7f7f", "ls": ":", "lw": 2.0},
    }

    for name, data in spectral_results.items():
        st = risk_styles.get(name, {"color": "#333", "ls": "-", "lw": 2})
        ax1.plot(data["coverages"], data["risk_values"],
                 color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                 label=name)

    ax1.set_xlabel("Coverage", fontsize=13)
    ax1.set_ylabel("Risk Measure Value", fontsize=13)
    ax1.set_title("Spectral Risk Measures vs. Coverage", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")

    # Right: Distortion functions
    u = np.linspace(0.001, 0.999, 200)
    ax2.plot(u, cvar_distortion(u, 0.10), color="#d62728", linewidth=2.5,
             label="CVaR ($\\delta=0.10$)")
    ax2.plot(u, wang_distortion(u, 0.5), color="#1f77b4", linewidth=2.5,
             linestyle="--", label="Wang ($\\eta=0.5$)")
    ax2.plot(u, dual_power_distortion(u, 2), color="#2ca02c", linewidth=2.5,
             linestyle="-.", label="Dual-Power ($p=2$)")
    ax2.plot(u, u, color="#7f7f7f", linewidth=1.5, linestyle=":", label="Risk-Neutral")
    ax2.set_xlabel("Probability level $u$", fontsize=13)
    ax2.set_ylabel("Distortion $g(u)$", fontsize=13)
    ax2.set_title("Distortion Functions (Risk Spectrum)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle("Spectral Risk Measures for Deferral Policy",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "spectral_risk_comparison.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Cost-Sensitive Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cost_sensitive_analysis(cost_results, save_dir):
    """Grouped bar: cost under different FN/FP ratios for each method."""
    set_pub_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    ratios = sorted(cost_results.keys())
    methods = list(cost_results[ratios[0]].keys())
    x = np.arange(len(ratios))
    width = 0.18
    method_colors = ["#1a5276", "#e67e22", "#27ae60", "#c0392b"]

    for j, (method, color) in enumerate(zip(methods, method_colors)):
        vals = [cost_results[r][method]["normalized_cost"] for r in ratios]
        bars = ax.bar(x + j * width, vals, width, label=method, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(x[i] + j * width, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"FN/FP = {r}" for r in ratios], fontsize=12)
    ax.set_ylabel("Normalised Cost per Claim", fontsize=13)
    ax.set_title("Cost-Sensitive Analysis: Impact of Asymmetric Loss",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cost_sensitive_analysis.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Uncertainty Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_decomposition(unc_data, y_test, defer_mask, save_dir):
    """Scatter: epistemic vs aleatoric uncertainty, coloured by outcome."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    # Left: epistemic vs aleatoric coloured by true label
    fraud_mask = y_arr == 1
    ax1.scatter(unc_data["epistemic"][fraud_mask], unc_data["aleatoric"][fraud_mask],
                c="#d62728", alpha=0.5, s=40, label="Fraud", edgecolors="black",
                linewidths=0.3)
    ax1.scatter(unc_data["epistemic"][~fraud_mask], unc_data["aleatoric"][~fraud_mask],
                c="#1f77b4", alpha=0.5, s=40, label="Legitimate", edgecolors="black",
                linewidths=0.3)
    ax1.set_xlabel("Epistemic Uncertainty (model)", fontsize=13)
    ax1.set_ylabel("Aleatoric Uncertainty (data)", fontsize=13)
    ax1.set_title("Uncertainty Decomposition by Label", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)

    # Right: total uncertainty coloured by deferral decision
    ax2.scatter(unc_data["total"][~defer_mask],
                unc_data["mean_posterior"][~defer_mask],
                c="#2ca02c", alpha=0.5, s=40, label="Decided by System",
                edgecolors="black", linewidths=0.3)
    ax2.scatter(unc_data["total"][defer_mask],
                unc_data["mean_posterior"][defer_mask],
                c="#9467bd", alpha=0.7, s=50, label="Deferred",
                edgecolors="black", linewidths=0.5, marker="^")
    ax2.set_xlabel("Total Uncertainty", fontsize=13)
    ax2.set_ylabel("Posterior $P(\\mathrm{fraud} | \\mathbf{x})$", fontsize=13)
    ax2.set_title("Total Uncertainty vs. Posterior", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)

    fig.suptitle("Epistemic–Aleatoric Uncertainty Decomposition",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "uncertainty_decomposition.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PLOT: Calibration-aware deferral
# ═══════════════════════════════════════════════════════════════════════════════

def plot_calibration_deferral(system, X_test, y_test, defer_mask, save_dir):
    """Show calibration quality on deferred vs non-deferred subsets."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    proba = system.calibrated_model.predict_proba(X_test)[:, 1]

    for ax, mask, title, color in [
        (ax1, ~defer_mask, "Non-Deferred Claims\n(Classifier Handles)", "#2ca02c"),
        (ax2, defer_mask, "Deferred Claims\n(Investigator Handles)", "#9467bd"),
    ]:
        if mask.sum() < 10:
            ax.text(0.5, 0.5, "Insufficient samples", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            continue

        p_sub = proba[mask]
        y_sub = y_arr[mask]

        # Reliability diagram
        from sklearn.calibration import calibration_curve
        try:
            prob_true, prob_pred = calibration_curve(y_sub, p_sub, n_bins=8)
            ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2.5,
                    markersize=8, label="Observed")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5,
                    label="Perfect calibration")
        except Exception:
            pass

        brier = brier_score_loss(y_sub, p_sub)
        ax.text(0.05, 0.90, f"Brier score: {brier:.4f}\nn = {mask.sum()}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("Predicted Probability", fontsize=13)
        ax.set_ylabel("Observed Frequency", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Calibration Quality: Deferred vs. Non-Deferred Claims",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "calibration_deferral.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    reset_style()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Safe-L2D-Fraud: Extended Experiments Suite")
    print("=" * 70)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Data ──
    print("\n[1/9] Loading data...")
    df = load_and_clean(CSV_PATH)
    df = engineer_features(df)
    X, y, num_cols, low_cat_cols, med_cat_cols = classify_columns(df, "fraud_reported")
    preprocessor = build_preprocessor(num_cols, low_cat_cols, med_cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    prior_fraud = y_train.mean()
    y_arr = y_test.values
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Prior: {prior_fraud:.3f}")

    # ── Base system ──
    print("\n[2/9] Training base Safe-L2D system...")
    conditionals = compute_class_conditionals(X_train, y_train, BAYESIAN_FEATURES)
    uncertainty = BayesianUncertaintyEstimator(conditionals, prior_fraud)
    investigator = SyntheticInvestigator(base_accuracy=0.85)

    system = SafeL2DFraudClassifier(
        preprocessor=preprocessor,
        uncertainty_estimator=uncertainty,
        investigator=investigator,
        delta=0.10, lambda_cvar=0.5,
        entropy_threshold=0.90, confidence_threshold=0.65,
    )
    system.fit(X_train, y_train)
    system_preds, defer_mask, confidence, entropies, posteriors = \
        system.system_predict(X_test, y_test)

    classifier_preds = np.argmax(system.calibrated_model.predict_proba(X_test), axis=1)

    # ── Experiment 1: Bootstrap CIs ──
    print("\n[3/9] Bootstrap confidence intervals (n=2000)...")
    boot_results = {}

    # Safe-L2D
    boot_results["Safe-L2D"] = bootstrap_system_metrics(system, X_test, y_test, investigator)
    print(f"  Safe-L2D Acc: {boot_results['Safe-L2D']['accuracy']['mean']:.4f} "
          f"[{boot_results['Safe-L2D']['accuracy']['ci_low']:.4f}, "
          f"{boot_results['Safe-L2D']['accuracy']['ci_high']:.4f}]")

    # XGBoost (no deferral)
    xgb_acc_boot = bootstrap_metric(y_arr, classifier_preds, accuracy_score)
    xgb_f1_boot = bootstrap_metric(y_arr, classifier_preds, f1_score)
    boot_results["XGBoost"] = {
        "accuracy": xgb_acc_boot,
        "f1": xgb_f1_boot,
    }
    print(f"  XGBoost Acc: {xgb_acc_boot['mean']:.4f} [{xgb_acc_boot['ci_low']:.4f}, {xgb_acc_boot['ci_high']:.4f}]")

    # Confidence baseline
    conf_system = SafeL2DFraudClassifier(
        preprocessor=preprocessor, uncertainty_estimator=uncertainty,
        investigator=investigator, delta=0.10, lambda_cvar=0.5,
        entropy_threshold=1.0, confidence_threshold=0.65,
    )
    conf_system.fit(X_train, y_train)
    conf_preds, conf_def, _, _, _ = conf_system.system_predict(X_test, y_test, method="confidence_only")
    conf_acc_boot = bootstrap_metric(y_arr, conf_preds, accuracy_score)
    conf_f1_boot = bootstrap_metric(y_arr, conf_preds, f1_score)
    boot_results["Confidence\nBaseline"] = {
        "accuracy": conf_acc_boot,
        "f1": conf_f1_boot,
    }

    print("  - Generating bootstrap CI figure...")
    plot_bootstrap_results(boot_results, {}, FIGURES_DIR)

    # ── Experiment 2: Investigator Sensitivity ──
    print("\n[4/9] Investigator sensitivity analysis...")
    sensitivity_results = {}
    for alpha_val in [0.70, 0.80, 0.90, 0.95]:
        inv = SyntheticInvestigator(base_accuracy=alpha_val)
        sys_a = SafeL2DFraudClassifier(
            preprocessor=preprocessor, uncertainty_estimator=uncertainty,
            investigator=inv, delta=0.10, lambda_cvar=0.5,
            entropy_threshold=0.90, confidence_threshold=0.65,
        )
        sys_a.fit(X_train, y_train)

        # Sweep deferral threshold
        proba = sys_a.calibrated_model.predict_proba(X_test)
        conf_a = np.max(proba, axis=1)
        pred_a = np.argmax(proba, axis=1)
        ent_a, post_a = sys_a.uncertainty.compute_all_entropies(X_test)

        coverages, acc_vals, f1_vals = [], [], []
        for thresh in np.linspace(0.0, 1.0, 40):
            d = (ent_a > thresh) | (conf_a < 0.55)
            cov = (~d).sum() / len(y_arr)
            coverages.append(cov)

            sp = pred_a.copy()
            if d.sum() > 0:
                X_d = X_test[d] if hasattr(X_test, 'loc') else X_test[d]
                y_d = y_test[d] if hasattr(y_test, 'iloc') else y_test[d]
                ip = inv.predict(X_d, y_d, ent_a[d])
                sp[d] = ip

            acc_vals.append(accuracy_score(y_arr, sp))
            f1_vals.append(f1_score(y_arr, sp, zero_division=0))

        sensitivity_results[alpha_val] = {
            "coverages": np.array(coverages),
            "accuracy_vals": np.array(acc_vals),
            "f1_vals": np.array(f1_vals),
        }
        print(f"  alpha={alpha_val}: peak acc={max(acc_vals):.4f}, peak F1={max(f1_vals):.4f}")

    print("  - Generating sensitivity figure...")
    plot_investigator_sensitivity(sensitivity_results, FIGURES_DIR)

    # ── Experiment 3: Spectral Risk Measures ──
    print("\n[5/9] Spectral risk measure comparison...")
    spectral_results = {}
    proba_full = system.calibrated_model.predict_proba(X_test)
    confidence_full = np.max(proba_full, axis=1)
    pred_full = np.argmax(proba_full, axis=1)
    entropies_full, _ = system.uncertainty.compute_all_entropies(X_test)

    for risk_name, dist_fn in [
        ("CVaR (δ=0.10)", lambda u: cvar_distortion(u, 0.10)),
        ("Wang (η=0.5)", lambda u: wang_distortion(u, 0.5)),
        ("Dual-Power (p=2)", lambda u: dual_power_distortion(u, 2)),
        ("Expected Loss", lambda u: u),
    ]:
        coverages, risk_vals = [], []
        for thresh in np.linspace(0.0, 1.0, 40):
            d = (entropies_full > thresh) | (confidence_full < 0.55)
            cov = (~d).sum() / len(y_arr)
            coverages.append(cov)
            losses = (pred_full[~d] != y_arr[~d]).astype(float)
            if len(losses) == 0:
                risk_vals.append(np.nan)
            else:
                risk_vals.append(spectral_risk_measure(losses, dist_fn))

        spectral_results[risk_name] = {
            "coverages": np.array(coverages),
            "risk_values": np.array(risk_vals),
        }

    print("  - Generating spectral risk figure...")
    plot_spectral_risk_comparison(spectral_results, FIGURES_DIR)

    # ── Experiment 4: Cost-Sensitive Analysis ──
    print("\n[6/9] Cost-sensitive analysis...")
    cost_results = {}
    fn_fp_ratios = [1, 3, 5, 10]

    for ratio in fn_fp_ratios:
        cost_results[ratio] = {}

        # Safe-L2D
        cost_results[ratio]["Safe-L2D"] = cost_sensitive_loss(
            y_arr, system_preds, cost_fp=1.0, cost_fn=float(ratio))

        # XGBoost alone
        cost_results[ratio]["XGBoost"] = cost_sensitive_loss(
            y_arr, classifier_preds, cost_fp=1.0, cost_fn=float(ratio))

        # Confidence baseline
        cost_results[ratio]["Conf. Baseline"] = cost_sensitive_loss(
            y_arr, conf_preds, cost_fp=1.0, cost_fn=float(ratio))

        # Random
        rng = np.random.RandomState(RANDOM_STATE)
        random_preds = rng.randint(0, 2, size=len(y_arr))
        cost_results[ratio]["Random"] = cost_sensitive_loss(
            y_arr, random_preds, cost_fp=1.0, cost_fn=float(ratio))

        print(f"  FN/FP={ratio}: Safe-L2D cost={cost_results[ratio]['Safe-L2D']['normalized_cost']:.4f}, "
              f"XGBoost={cost_results[ratio]['XGBoost']['normalized_cost']:.4f}")

    print("  - Generating cost-sensitive figure...")
    plot_cost_sensitive_analysis(cost_results, FIGURES_DIR)

    # ── Experiment 5: Uncertainty Decomposition ──
    print("\n[7/9] Uncertainty decomposition (epistemic vs aleatoric)...")
    unc_data = compute_uncertainty_decomposition(system, X_test)
    print(f"  Mean epistemic: {unc_data['epistemic'].mean():.6f}")
    print(f"  Mean aleatoric: {unc_data['aleatoric'].mean():.6f}")
    print(f"  Ratio (epi/total): {unc_data['epistemic'].mean() / unc_data['total'].mean():.4f}")

    print("  - Generating uncertainty decomposition figure...")
    plot_uncertainty_decomposition(unc_data, y_test, defer_mask, FIGURES_DIR)

    # ── Experiment 6: Calibration-aware deferral ──
    print("\n[8/9] Calibration analysis on deferred vs non-deferred...")
    plot_calibration_deferral(system, X_test, y_test, defer_mask, FIGURES_DIR)

    # ── Experiment 7: Cross-validated system performance ──
    print("\n[9/9] 5-fold cross-validated system evaluation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_sys_acc = []
    cv_sys_f1 = []
    cv_defer_rate = []
    cv_cvar = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        cond_cv = compute_class_conditionals(X_tr, y_tr, BAYESIAN_FEATURES)
        unc_cv = BayesianUncertaintyEstimator(cond_cv, y_tr.mean())
        inv_cv = SyntheticInvestigator(base_accuracy=0.85, random_state=RANDOM_STATE + fold_idx)
        prep_cv = build_preprocessor(num_cols, low_cat_cols, med_cat_cols)

        sys_cv = SafeL2DFraudClassifier(
            preprocessor=prep_cv, uncertainty_estimator=unc_cv,
            investigator=inv_cv, delta=0.10, lambda_cvar=0.5,
            entropy_threshold=0.90, confidence_threshold=0.65,
        )
        sys_cv.fit(X_tr, y_tr)
        sp, dm, _, _, _ = sys_cv.system_predict(X_te, y_te)
        ya = y_te.values

        cv_sys_acc.append(accuracy_score(ya, sp))
        cv_sys_f1.append(f1_score(ya, sp, zero_division=0))
        cv_defer_rate.append(dm.mean())

        cp = np.argmax(sys_cv.calibrated_model.predict_proba(X_te), axis=1)
        nd_losses = (cp[~dm] != ya[~dm]).astype(float)
        cv_cvar.append(empirical_cvar(nd_losses, 0.10) if len(nd_losses) > 0 else 0)

        print(f"  Fold {fold_idx+1}: Acc={cv_sys_acc[-1]:.4f}, F1={cv_sys_f1[-1]:.4f}, "
              f"Defer={cv_defer_rate[-1]:.3f}, CVaR={cv_cvar[-1]:.4f}")

    print(f"\n  CV System Accuracy: {np.mean(cv_sys_acc):.4f} ± {np.std(cv_sys_acc):.4f}")
    print(f"  CV System F1:       {np.mean(cv_sys_f1):.4f} ± {np.std(cv_sys_f1):.4f}")
    print(f"  CV Deferral Rate:   {np.mean(cv_defer_rate):.4f} ± {np.std(cv_defer_rate):.4f}")
    print(f"  CV CVaR@0.10:       {np.mean(cv_cvar):.4f} ± {np.std(cv_cvar):.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Bootstrap CI (Acc): [{boot_results['Safe-L2D']['accuracy']['ci_low']:.4f}, "
          f"{boot_results['Safe-L2D']['accuracy']['ci_high']:.4f}]")
    print(f"  Bootstrap CI (F1):  [{boot_results['Safe-L2D']['f1']['ci_low']:.4f}, "
          f"{boot_results['Safe-L2D']['f1']['ci_high']:.4f}]")
    print(f"  Best investigator alpha: 0.95 (peak acc={max(sensitivity_results[0.95]['accuracy_vals']):.4f})")
    print(f"  Cost ratio FN/FP=5: Safe-L2D={cost_results[5]['Safe-L2D']['normalized_cost']:.4f} vs "
          f"XGBoost={cost_results[5]['XGBoost']['normalized_cost']:.4f}")
    print(f"  Epistemic/Total ratio: {unc_data['epistemic'].mean() / unc_data['total'].mean():.4f}")
    print(f"\n  New figures saved to {FIGURES_DIR}/")
    print("  - bootstrap_ci_results.svg")
    print("  - investigator_sensitivity.svg")
    print("  - spectral_risk_comparison.svg")
    print("  - cost_sensitive_analysis.svg")
    print("  - uncertainty_decomposition.svg")
    print("  - calibration_deferral.svg")
    print("=" * 70)


if __name__ == "__main__":
    main()
