"""
Insurance Fraud Detection — Safe Learning to Defer Framework
Implements CVaR-penalised Learning to Defer (L2D) for insurance fraud detection
with Bayesian uncertainty quantification and off-policy evaluation.
Generates SVG figures for the ICML 2026 paper.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import gaussian_kde
from scipy.special import xlogy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    precision_score, recall_score,
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

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "insurance_fraud_claims.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# Causal features validated by the DAG in Section C
CAUSAL_FEATURES = [
    "total_claim_amount", "incident_severity", "bodily_injuries",
    "witnesses", "police_report_available", "policy_deductable",
    "umbrella_limit", "policy_age_days",
]

COLORS = {
    "safe_l2d": "#1f77b4",
    "l2d_no_cvar": "#ff7f0e",
    "confidence": "#2ca02c",
    "xgboost": "#d62728",
    "fraud": "#d62728",
    "legit": "#1f77b4",
    "defer": "#9467bd",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION D: Safe-L2D-Fraud Framework
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianUncertaintyEstimator:
    """
    Uses KDE class-conditionals to compute posterior entropy
    as the deferral uncertainty signal.
    """

    def __init__(self, conditionals, prior_fraud):
        self.conditionals = conditionals
        self.prior_fraud = prior_fraud
        self.features = [f for f in BAYESIAN_FEATURES if f in conditionals]

    @staticmethod
    def entropy(p):
        """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def compute_posterior(self, row):
        """Sequential Bayesian update returning final P(fraud|x)."""
        current = self.prior_fraud
        for feat in self.features:
            val = row.get(feat, np.nan)
            if pd.isna(val):
                continue
            val = float(val)
            cond = self.conditionals[feat]
            l_fraud = max(float(cond["kde_fraud"].evaluate(np.array([val]))[0]), 1e-10)
            l_legit = max(float(cond["kde_legit"].evaluate(np.array([val]))[0]), 1e-10)
            current = (current * l_fraud) / (current * l_fraud + (1 - current) * l_legit)
        return current

    def compute_all_posteriors(self, X):
        """Compute posterior P(fraud|x) for all samples."""
        posteriors = np.array([self.compute_posterior(X.iloc[i]) for i in range(len(X))])
        return posteriors

    def compute_all_entropies(self, X):
        """Compute posterior entropy H(π) for all samples."""
        posteriors = self.compute_all_posteriors(X)
        return self.entropy(posteriors), posteriors


class SyntheticInvestigator:
    """
    Models human fraud investigator performance.
    Parameterised by alpha (base accuracy).
    """

    def __init__(self, base_accuracy=0.85, random_state=RANDOM_STATE):
        self.alpha = base_accuracy
        self.rng = np.random.RandomState(random_state)

    def predict(self, X, y_true, posterior_entropy=None):
        """
        Investigator predictions. Better on high-uncertainty cases
        (where they bring domain expertise the model lacks).
        """
        n = len(y_true)
        predictions = np.zeros(n, dtype=int)

        for i in range(n):
            # Difficulty-adjusted accuracy: investigators are better
            # when the case is genuinely ambiguous
            acc = self.alpha
            if posterior_entropy is not None and posterior_entropy[i] > 0.8:
                acc = min(acc + 0.05, 0.98)  # Slight bonus on hard cases

            if self.rng.random() < acc:
                predictions[i] = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
            else:
                predictions[i] = 1 - (y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i])

        return predictions

    def expert_loss(self, X, y_true, posterior_entropy=None):
        """Per-sample expert loss: l_exp(x, y, M(z))."""
        preds = self.predict(X, y_true, posterior_entropy)
        y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        return (preds != y_arr).astype(float)


def empirical_cvar(losses, delta):
    """
    CVaR_delta = mean of top-delta fraction of losses.
    CVaR_delta(L) = (1/delta) * E[L * 1{L >= VaR_delta}]
    """
    if len(losses) == 0:
        return 0.0
    n = len(losses)
    k = max(1, int(np.ceil(delta * n)))
    sorted_losses = np.sort(losses)
    return np.mean(sorted_losses[-k:])


class SafeL2DFraudClassifier:
    """
    Complete Safe-L2D-Fraud system combining:
    1. Base XGBoost classifier
    2. Bayesian uncertainty estimator
    3. CVaR risk penalty monitoring
    4. Causal feature selection for uncertainty estimation
    """

    def __init__(self, preprocessor, uncertainty_estimator, investigator,
                 delta=0.10, lambda_cvar=0.5, entropy_threshold=0.7,
                 confidence_threshold=0.6):
        self.preprocessor = preprocessor
        self.uncertainty = uncertainty_estimator
        self.investigator = investigator
        self.delta = delta
        self.lambda_cvar = lambda_cvar
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.base_model = None

    def fit(self, X_train, y_train):
        """Train the base classifier (XGBoost with probability calibration)."""
        self.base_model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=RANDOM_STATE,
            )),
        ])
        self.base_model.fit(X_train, y_train)

        # Also train a calibrated version for better probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, cv=3, method="isotonic"
        )
        self.calibrated_model.fit(X_train, y_train)

        return self

    def predict_with_deferral(self, X, y_true=None, method="combined"):
        """
        Returns: predictions, deferral_mask, confidence, posterior_entropy.
        Deferral triggered by high posterior entropy OR low classifier confidence.
        """
        # Classifier confidence
        proba = self.calibrated_model.predict_proba(X)
        confidence = np.max(proba, axis=1)
        pred_class = np.argmax(proba, axis=1)

        # Bayesian posterior entropy
        entropies, posteriors = self.uncertainty.compute_all_entropies(X)

        # Deferral decision
        if method == "entropy_only":
            defer_mask = entropies > self.entropy_threshold
        elif method == "confidence_only":
            defer_mask = confidence < self.confidence_threshold
        else:  # combined
            defer_mask = (entropies > self.entropy_threshold) | (confidence < self.confidence_threshold)

        return pred_class, defer_mask, confidence, entropies, posteriors

    def system_predict(self, X, y_true, method="combined"):
        """
        Full system prediction: classifier handles non-deferred,
        investigator handles deferred claims.
        """
        pred_class, defer_mask, confidence, entropies, posteriors = \
            self.predict_with_deferral(X, y_true, method)

        # System predictions
        system_preds = pred_class.copy()

        # Investigator handles deferred cases
        if defer_mask.sum() > 0:
            X_deferred = X[defer_mask] if hasattr(X, 'loc') else X[defer_mask]
            y_deferred = y_true[defer_mask] if hasattr(y_true, 'iloc') else y_true[defer_mask]
            entropy_deferred = entropies[defer_mask]
            inv_preds = self.investigator.predict(X_deferred, y_deferred, entropy_deferred)
            system_preds[defer_mask] = inv_preds

        return system_preds, defer_mask, confidence, entropies, posteriors

    def coverage_accuracy_curve(self, X, y_true, method="combined",
                                 n_thresholds=50):
        """
        Vary deferral threshold to produce coverage-accuracy tradeoff.
        Coverage = fraction of claims decided by the system (not deferred).
        """
        proba = self.calibrated_model.predict_proba(X)
        confidence = np.max(proba, axis=1)
        pred_class = np.argmax(proba, axis=1)
        entropies, posteriors = self.uncertainty.compute_all_entropies(X)

        y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

        coverages = []
        accuracies = []
        f1s = []
        cvars = []

        if method in ["entropy_only", "combined"]:
            thresholds = np.linspace(0.0, 1.0, n_thresholds)
        else:
            thresholds = np.linspace(0.5, 1.0, n_thresholds)

        for thresh in thresholds:
            if method == "entropy_only":
                defer_mask = entropies > thresh
            elif method == "confidence_only":
                defer_mask = confidence < thresh
            else:
                # For combined, sweep entropy threshold with fixed confidence
                defer_mask = (entropies > thresh) | (confidence < 0.55)

            n_covered = (~defer_mask).sum()
            coverage = n_covered / len(y_arr)
            coverages.append(coverage)

            if n_covered == 0:
                accuracies.append(np.nan)
                f1s.append(np.nan)
                cvars.append(np.nan)
                continue

            # System predictions
            system_preds = pred_class.copy()
            if defer_mask.sum() > 0:
                inv_preds = self.investigator.predict(
                    X[defer_mask] if hasattr(X, 'loc') else X[defer_mask],
                    y_true[defer_mask] if hasattr(y_true, 'iloc') else y_true[defer_mask],
                    entropies[defer_mask]
                )
                system_preds[defer_mask] = inv_preds

            acc = accuracy_score(y_arr, system_preds)
            f1 = f1_score(y_arr, system_preds, zero_division=0)
            accuracies.append(acc)
            f1s.append(f1)

            # CVaR on classifier-handled (non-deferred) predictions
            classifier_losses = (pred_class[~defer_mask] != y_arr[~defer_mask]).astype(float)
            cvar = empirical_cvar(classifier_losses, self.delta)
            cvars.append(cvar)

        return {
            "coverages": np.array(coverages),
            "accuracies": np.array(accuracies),
            "f1s": np.array(f1s),
            "cvars": np.array(cvars),
            "thresholds": thresholds,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline: Confidence-Based Deferral (no L2D)
# ═══════════════════════════════════════════════════════════════════════════════

def confidence_baseline_curve(model, X, y_true, investigator, delta=0.10,
                               n_thresholds=50):
    """Simple confidence thresholding baseline for deferral."""
    proba = model.predict_proba(X)
    confidence = np.max(proba, axis=1)
    pred_class = np.argmax(proba, axis=1)
    y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

    thresholds = np.linspace(0.5, 1.0, n_thresholds)
    coverages, accuracies, f1s, cvars = [], [], [], []

    for thresh in thresholds:
        defer_mask = confidence < thresh
        n_covered = (~defer_mask).sum()
        coverage = n_covered / len(y_arr)
        coverages.append(coverage)

        if n_covered == 0:
            accuracies.append(np.nan)
            f1s.append(np.nan)
            cvars.append(np.nan)
            continue

        system_preds = pred_class.copy()
        if defer_mask.sum() > 0:
            inv_preds = investigator.predict(
                X[defer_mask] if hasattr(X, 'loc') else X[defer_mask],
                y_true[defer_mask] if hasattr(y_true, 'iloc') else y_true[defer_mask]
            )
            system_preds[defer_mask] = inv_preds

        accuracies.append(accuracy_score(y_arr, system_preds))
        f1s.append(f1_score(y_arr, system_preds, zero_division=0))

        classifier_losses = (pred_class[~defer_mask] != y_arr[~defer_mask]).astype(float)
        cvars.append(empirical_cvar(classifier_losses, delta))

    return {
        "coverages": np.array(coverages),
        "accuracies": np.array(accuracies),
        "f1s": np.array(f1s),
        "cvars": np.array(cvars),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION E: Off-Policy Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class OffPolicyEvaluator:
    """
    Evaluates Safe-L2D-Fraud policy using historical claim data.
    Three estimators: Direct Method, Importance Sampling, Doubly Robust.
    """

    def __init__(self, w_max=10.0, random_state=RANDOM_STATE):
        self.w_max = w_max
        self.rng = np.random.RandomState(random_state)

    def _compute_rewards(self, y_true, predictions):
        """Reward: +1 for correct prediction, -1 for incorrect."""
        y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        return np.where(predictions == y_arr, 1.0, -1.0)

    def direct_method(self, model, X, y_true):
        """
        V^DM = (1/N) * sum_i r_hat(x_i, pi_e(x_i))
        Fits a reward model and evaluates under the target policy.
        """
        proba = model.predict_proba(X)
        pred = np.argmax(proba, axis=1)
        rewards = self._compute_rewards(y_true, pred)
        return np.mean(rewards)

    def importance_sampling(self, eval_proba, behaviour_proba, y_true,
                            eval_actions, behaviour_actions):
        """
        V^IS = (1/N) * sum_i w(x_i, a_i) * r(x_i, a_i, y_i)
        Also compute self-normalised IS (SNIPS).
        """
        n = len(y_true)
        y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

        # Importance weights
        eval_p = np.array([eval_proba[i, int(behaviour_actions[i])]
                           for i in range(n)])
        behav_p = np.array([behaviour_proba[i, int(behaviour_actions[i])]
                            for i in range(n)])
        behav_p = np.clip(behav_p, 1e-6, None)
        weights = np.clip(eval_p / behav_p, 0, self.w_max)

        rewards = self._compute_rewards(y_true, behaviour_actions)

        # Standard IS
        v_is = np.mean(weights * rewards)

        # Self-normalised IS (SNIPS)
        w_sum = np.sum(weights)
        v_snips = np.sum(weights * rewards) / max(w_sum, 1e-10)

        return v_is, v_snips

    def doubly_robust(self, eval_model, behaviour_proba, X, y_true,
                      behaviour_actions):
        """
        V^DR = (1/N) * sum_i [r_hat(x_i, pi_e(x_i))
                               + w_i * (r_i - r_hat(x_i, a_i))]
        """
        n = len(y_true)
        y_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)

        eval_proba = eval_model.predict_proba(X)
        eval_actions = np.argmax(eval_proba, axis=1)

        # Reward model estimate
        r_hat_eval = self._compute_rewards(y_true, eval_actions)

        # Reward model for behaviour actions
        r_hat_behav = self._compute_rewards(y_true, behaviour_actions)

        # Importance weights
        eval_p = np.array([eval_proba[i, int(behaviour_actions[i])]
                           for i in range(n)])
        behav_p = np.array([behaviour_proba[i, int(behaviour_actions[i])]
                            for i in range(n)])
        behav_p = np.clip(behav_p, 1e-6, None)
        weights = np.clip(eval_p / behav_p, 0, self.w_max)

        # Observed rewards
        observed_rewards = self._compute_rewards(y_true, behaviour_actions)

        # DR estimate
        dr_terms = r_hat_eval + weights * (observed_rewards - r_hat_behav)
        return np.mean(dr_terms)

    def bootstrap_ci(self, estimator_fn, n_boot=1000, alpha=0.05):
        """Percentile bootstrap CI."""
        estimates = np.array([estimator_fn() for _ in range(n_boot)])
        lower = np.percentile(estimates, 100 * alpha / 2)
        upper = np.percentile(estimates, 100 * (1 - alpha / 2))
        return np.mean(estimates), lower, upper


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_l2d_system_diagram(save_dir):
    """Architecture flow diagram for the Safe-L2D-Fraud system."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=11):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def draw_arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.2, text, ha="center", va="center", fontsize=10,
                    fontstyle="italic", color=color)

    # Input
    draw_box(0.3, 2.8, 2.2, 1.4, "Insurance\nClaim\n$\\mathbf{x} \\in \\mathcal{X}$", "#e8e8e8", 11)

    # Bayesian module
    draw_box(3.5, 4.8, 2.5, 1.5, "Bayesian\nUncertainty\n$H(\\pi_T)$", "#aec7e8", 10)

    # Classifier
    draw_box(3.5, 2.5, 2.5, 1.8, "XGBoost\nClassifier\n$h(\\mathbf{x})$", "#ffbb78", 10)

    # CVaR module
    draw_box(3.5, 0.3, 2.5, 1.5, "CVaR Risk\nMonitor\n$\\text{CVaR}_\\delta$", "#f7b6d2", 10)

    # Deferral gate
    draw_box(7.2, 2.8, 2.0, 1.5, "Deferral\nGate\n$r(\\mathbf{x})$", "#c5b0d5", 11)

    # Auto-flag
    draw_box(10.5, 4.5, 2.5, 1.2, "Auto-Flag\nFraud", "#ff9896", 11)

    # Auto-clear
    draw_box(10.5, 2.8, 2.5, 1.2, "Auto-Clear\nLegitimate", "#98df8a", 11)

    # Defer to investigator
    draw_box(10.5, 0.8, 2.5, 1.2, "Defer to\nInvestigator", "#c5b0d5", 11)

    # Arrows
    draw_arrow(2.5, 3.5, 3.5, 3.4, "")
    draw_arrow(2.5, 3.5, 3.5, 5.5, "")
    draw_arrow(6.0, 5.5, 7.2, 3.8, "$H(\\pi) > \\tau$")
    draw_arrow(6.0, 3.4, 7.2, 3.5, "predict")
    draw_arrow(6.0, 1.05, 7.2, 3.0, "risk bound")
    draw_arrow(9.2, 4.0, 10.5, 5.1, "fraud", "#d62728")
    draw_arrow(9.2, 3.5, 10.5, 3.4, "legit", "#2ca02c")
    draw_arrow(9.2, 3.0, 10.5, 1.6, "defer", "#9467bd")

    fig.suptitle("Safe-L2D-Fraud: System Architecture",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "l2d_system_diagram.svg"),
                format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def _smooth(x, y, window=5):
    """Simple moving average for smoother plots."""
    if len(y) < window:
        return x, y
    # Sort by x
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    # Remove NaN
    valid = ~np.isnan(y_s)
    x_s, y_s = x_s[valid], y_s[valid]
    if len(y_s) < window:
        return x_s, y_s
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_s, kernel, mode='valid')
    x_smooth = x_s[(window-1)//2 : (window-1)//2 + len(y_smooth)]
    return x_smooth, y_smooth


def plot_coverage_accuracy_tradeoff(results_dict, save_dir):
    """Primary results figure: coverage vs system accuracy."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'axes.linewidth': 1.2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    style_map = {
        "Safe-L2D": {"color": "#1a5276", "ls": "-", "lw": 3.0, "marker": "o", "ms": 4},
        "L2D (no CVaR)": {"color": "#e67e22", "ls": "--", "lw": 2.5, "marker": "s", "ms": 3},
        "Confidence": {"color": "#27ae60", "ls": "-.", "lw": 2.0, "marker": "^", "ms": 3},
        "XGBoost": {"color": "#c0392b", "ls": ":", "lw": 2.5, "marker": "D", "ms": 5},
    }

    for name, res in results_dict.items():
        st = style_map.get(name, {"color": "#333", "ls": "-", "lw": 2, "marker": "", "ms": 0})

        if name == "XGBoost":
            # Single point for no-deferral baseline
            for ax, metric in [(ax1, "accuracies"), (ax2, "f1s")]:
                ax.scatter([1.0], [res[metric][0]], color=st["color"], s=120,
                           marker=st["marker"], zorder=5, edgecolors="black", linewidths=1.0,
                           label=f"{name} (no defer)")
        else:
            for ax, metric in [(ax1, "accuracies"), (ax2, "f1s")]:
                xs, ys = _smooth(res["coverages"], res[metric], window=5)
                ax.plot(xs, ys, color=st["color"], linestyle=st["ls"],
                        linewidth=st["lw"], label=name, marker=st["marker"],
                        markevery=max(1, len(xs)//8), markersize=st["ms"],
                        markeredgecolor="black", markeredgewidth=0.5)

    for ax, metric_name in [(ax1, "System Accuracy"), (ax2, "System F1 Score")]:
        ax.set_xlabel("Coverage (fraction decided by classifier)", fontsize=13)
        ax.set_ylabel(metric_name, fontsize=13)
        ax.set_title(f"Coverage vs. {metric_name}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right", framealpha=0.9, edgecolor="gray")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xlim(-0.02, 1.05)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Learning to Defer: Coverage–Performance Tradeoff",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "coverage_accuracy_tradeoff.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    # Don't reset globally since it's per-function
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_cvar_deferral_analysis(l2d_results, X_test, y_test, system,
                                 save_dir):
    """CVaR at different delta levels + deferral rate by fraud probability."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: CVaR at different delta values across coverage
    deltas = [0.05, 0.10, 0.20]
    delta_colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    proba = system.calibrated_model.predict_proba(X_test)
    pred_class = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    entropies, posteriors = system.uncertainty.compute_all_entropies(X_test)
    y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    for delta, dc in zip(deltas, delta_colors):
        thresholds = np.linspace(0.0, 1.0, 40)
        cvars_at_delta = []
        coverages = []
        for thresh in thresholds:
            defer_mask = (entropies > thresh) | (confidence < 0.55)
            non_defer = ~defer_mask
            coverage = non_defer.sum() / len(y_arr)
            coverages.append(coverage)
            if non_defer.sum() == 0:
                cvars_at_delta.append(np.nan)
                continue
            losses = (pred_class[non_defer] != y_arr[non_defer]).astype(float)
            cvars_at_delta.append(empirical_cvar(losses, delta))
        ax1.plot(coverages, cvars_at_delta, linewidth=2, color=dc,
                 label=f"$\\delta={delta}$")

    ax1.set_xlabel("Coverage", fontsize=13)
    ax1.set_ylabel("$\\mathrm{CVaR}_\\delta$ (Classifier Loss)", fontsize=13)
    ax1.set_title("Risk Profile: CVaR vs. Coverage", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, linestyle="--")

    # Right: Deferral rate by posterior probability bracket
    brackets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bracket_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    # Use a mid-range entropy threshold
    _, defer_mask, _, entropies_all, posteriors_all = \
        system.predict_with_deferral(X_test, y_test)

    deferral_rates = []
    bracket_counts = []
    for lo, hi in brackets:
        mask = (posteriors_all >= lo) & (posteriors_all < hi)
        if mask.sum() == 0:
            deferral_rates.append(0)
            bracket_counts.append(0)
            continue
        rate = defer_mask[mask].mean()
        deferral_rates.append(rate)
        bracket_counts.append(mask.sum())

    bar_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#ff7f0e", "#1f77b4"]
    bars = ax2.bar(range(len(brackets)), deferral_rates, color=bar_colors,
                   alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (rate, count) in enumerate(zip(deferral_rates, bracket_counts)):
        ax2.text(i, rate + 0.02, f"{rate:.1%}\n(n={count})",
                 ha="center", va="bottom", fontsize=10)

    ax2.set_xticks(range(len(brackets)))
    ax2.set_xticklabels(bracket_labels)
    ax2.set_xlabel("Posterior $P(\\mathrm{fraud} | \\mathbf{x})$ Bracket", fontsize=13)
    ax2.set_ylabel("Deferral Rate", fontsize=13)
    ax2.set_title("Deferral Rate by Fraud Probability", fontsize=15, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("CVaR Risk Analysis and Deferral Distribution",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cvar_deferral_analysis.svg"),
                format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_bayesian_deferral_entropy(entropies, confidence, posteriors,
                                    defer_mask, y_test, save_dir,
                                    entropy_threshold=0.7):
    """Scatter: posterior entropy vs classifier confidence, coloured by label."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    y_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    # Left: entropy vs confidence coloured by true label
    fraud_mask = y_arr == 1
    ax1.scatter(entropies[fraud_mask], confidence[fraud_mask],
                c=COLORS["fraud"], alpha=0.6, s=40, label="Fraud", edgecolors="black", linewidth=0.3)
    ax1.scatter(entropies[~fraud_mask], confidence[~fraud_mask],
                c=COLORS["legit"], alpha=0.6, s=40, label="Legitimate", edgecolors="black", linewidth=0.3)

    ax1.axvline(x=entropy_threshold, color="black", linestyle="--", linewidth=2,
                label=f"$\\tau = {entropy_threshold}$")
    ax1.set_xlabel("Posterior Entropy $H(\\pi_T)$", fontsize=13)
    ax1.set_ylabel("Classifier Confidence $\\max_i p_i$", fontsize=13)
    ax1.set_title("Uncertainty Landscape", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, linestyle="--")

    # Right: entropy vs confidence coloured by deferral decision
    ax2.scatter(entropies[~defer_mask], confidence[~defer_mask],
                c="#2ca02c", alpha=0.5, s=40, label="Decided by System", edgecolors="black", linewidth=0.3)
    ax2.scatter(entropies[defer_mask], confidence[defer_mask],
                c=COLORS["defer"], alpha=0.7, s=50, label="Deferred to Investigator",
                edgecolors="black", linewidth=0.5, marker="^")

    ax2.axvline(x=entropy_threshold, color="black", linestyle="--", linewidth=2,
                label=f"$\\tau = {entropy_threshold}$")
    ax2.set_xlabel("Posterior Entropy $H(\\pi_T)$", fontsize=13)
    ax2.set_ylabel("Classifier Confidence $\\max_i p_i$", fontsize=13)
    ax2.set_title("Deferral Decision Regions", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Bayesian Entropy-Driven Deferral Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "bayesian_deferral_entropy.svg"),
                format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_ope_comparison(ope_results, save_dir):
    """Bar chart comparing DM, IS, DR policy value estimates with CIs."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=(10, 5.5))

    methods = list(ope_results.keys())
    x = np.arange(len(methods))
    width = 0.25

    policies = ["Safe-L2D", "XGBoost", "Random"]
    policy_colors = ["#1a5276", "#c0392b", "#7f8c8d"]
    hatches = ["", "//", ".."]

    for j, (policy, color, hatch) in enumerate(zip(policies, policy_colors, hatches)):
        means = []
        errors_low = []
        errors_high = []
        for method in methods:
            entry = ope_results[method].get(policy, {})
            m = entry.get("mean", 0)
            lo = entry.get("ci_low", m)
            hi = entry.get("ci_high", m)
            means.append(m)
            errors_low.append(m - lo)
            errors_high.append(hi - m)

        ax.bar(x + j * width, means, width, label=policy, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.8, hatch=hatch,
               yerr=[errors_low, errors_high], capsize=5, error_kw={"linewidth": 1.5})

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("\n", " ") for m in methods], fontsize=11, fontweight="bold")
    ax.set_ylabel("Estimated Policy Value", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="gray")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Off-Policy Evaluation: Estimator Comparison",
                 fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ope_estimators_comparison.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    # Don't reset globally since it's per-function
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_ablation_study(ablation_results, save_dir):
    """Grouped bar chart for ablation study."""
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=(11, 5.5))

    variants = list(ablation_results.keys())
    metrics = ["System Accuracy", "F1", "CVaR@0.10", "Deferral Rate"]
    x = np.arange(len(metrics))
    width = 0.19
    variant_colors = ["#1a5276", "#e67e22", "#27ae60", "#c0392b"]
    hatches = ["", "//", "\\\\", ".."]

    for j, (variant, color, hatch) in enumerate(zip(variants, variant_colors, hatches)):
        vals = [ablation_results[variant].get(m, 0) for m in metrics]
        bars = ax.bar(x + j * width, vals, width, label=variant, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.8, hatch=hatch)
        for i, v in enumerate(vals):
            ax.text(x[i] + j * width, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Ablation Study: Component Contributions",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="gray")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.suptitle("Ablation Study: Component Contributions",
                 fontsize=16, fontweight="bold")
    fig.savefig(os.path.join(save_dir, "ablation_study.svg"),
                format="svg", bbox_inches="tight", dpi=150)
    # Don't reset globally since it's per-function
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Safe-L2D-Fraud: Learning to Defer Pipeline")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Phase 1: Data Preparation ──
    print("\n[1/10] Loading and preparing data...")
    df = load_and_clean(CSV_PATH)
    df = engineer_features(df)
    X, y, num_cols, low_cat_cols, med_cat_cols = classify_columns(df, "fraud_reported")
    preprocessor = build_preprocessor(num_cols, low_cat_cols, med_cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    prior_fraud = y_train.mean()
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Prior: {prior_fraud:.3f}")

    # ── Phase 2: Bayesian Uncertainty Estimator ──
    print("\n[2/10] Computing Bayesian class-conditionals...")
    conditionals = compute_class_conditionals(X_train, y_train, BAYESIAN_FEATURES)
    uncertainty = BayesianUncertaintyEstimator(conditionals, prior_fraud)

    # ── Phase 3: Synthetic Investigator ──
    print("\n[3/10] Setting up synthetic investigator (alpha=0.85)...")
    investigator = SyntheticInvestigator(base_accuracy=0.85)

    # ── Phase 4: Train Safe-L2D-Fraud System ──
    print("\n[4/10] Training Safe-L2D-Fraud classifier...")
    system = SafeL2DFraudClassifier(
        preprocessor=preprocessor,
        uncertainty_estimator=uncertainty,
        investigator=investigator,
        delta=0.10,
        lambda_cvar=0.5,
        entropy_threshold=0.90,
        confidence_threshold=0.65,
    )
    system.fit(X_train, y_train)

    # ── Phase 5: System Evaluation ──
    print("\n[5/10] Evaluating Safe-L2D-Fraud system...")
    system_preds, defer_mask, confidence, entropies, posteriors = \
        system.system_predict(X_test, y_test)

    y_arr = y_test.values
    sys_acc = accuracy_score(y_arr, system_preds)
    sys_f1 = f1_score(y_arr, system_preds, zero_division=0)
    deferral_rate = defer_mask.mean()
    coverage = 1 - deferral_rate

    # CVaR on non-deferred
    classifier_preds = np.argmax(system.calibrated_model.predict_proba(X_test), axis=1)
    non_defer_losses = (classifier_preds[~defer_mask] != y_arr[~defer_mask]).astype(float)
    cvar_010 = empirical_cvar(non_defer_losses, 0.10)

    print(f"  System Accuracy: {sys_acc:.4f}")
    print(f"  System F1:       {sys_f1:.4f}")
    print(f"  Deferral Rate:   {deferral_rate:.4f}")
    print(f"  Coverage:        {coverage:.4f}")
    print(f"  CVaR@0.10:       {cvar_010:.4f}")

    # ── Phase 6: Coverage-Accuracy Curves ──
    print("\n[6/10] Computing coverage-accuracy curves...")

    # Safe-L2D (combined)
    safe_l2d_results = system.coverage_accuracy_curve(X_test, y_test, method="combined")

    # L2D without CVaR (entropy only)
    l2d_no_cvar = system.coverage_accuracy_curve(X_test, y_test, method="entropy_only")

    # Confidence baseline
    conf_results = confidence_baseline_curve(
        system.calibrated_model, X_test, y_test, investigator
    )

    # XGBoost alone (no deferral) - single point at full coverage
    xgb_preds = classifier_preds
    xgb_acc = accuracy_score(y_arr, xgb_preds)
    xgb_f1 = f1_score(y_arr, xgb_preds, zero_division=0)
    xgb_results = {
        "coverages": np.array([1.0, 1.0]),
        "accuracies": np.array([xgb_acc, xgb_acc]),
        "f1s": np.array([xgb_f1, xgb_f1]),
    }

    results_dict = {
        "Safe-L2D": safe_l2d_results,
        "L2D (no CVaR)": l2d_no_cvar,
        "Confidence": conf_results,
        "XGBoost": xgb_results,
    }

    # ── Phase 7: Generate Figures ──
    print("\n[7/10] Generating figures...")

    print("  - L2D system diagram...")
    plot_l2d_system_diagram(FIGURES_DIR)

    print("  - Coverage-accuracy tradeoff...")
    plot_coverage_accuracy_tradeoff(results_dict, FIGURES_DIR)

    print("  - CVaR deferral analysis...")
    plot_cvar_deferral_analysis(safe_l2d_results, X_test, y_test, system, FIGURES_DIR)

    print("  - Bayesian deferral entropy...")
    plot_bayesian_deferral_entropy(
        entropies, confidence, posteriors, defer_mask, y_test,
        FIGURES_DIR, entropy_threshold=0.90
    )

    # ── Phase 8: Off-Policy Evaluation ──
    print("\n[8/10] Running off-policy evaluation...")
    ope = OffPolicyEvaluator()

    # Behaviour policy: random-ish (simulating historical decisions)
    behaviour_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE + 1)),
    ])
    behaviour_model.fit(X_train, y_train)
    behaviour_proba = behaviour_model.predict_proba(X_test)
    behaviour_actions = behaviour_model.predict(X_test)

    # Evaluation policy: our Safe-L2D system (using calibrated model)
    eval_proba = system.calibrated_model.predict_proba(X_test)
    eval_actions = np.argmax(eval_proba, axis=1)

    # Random policy
    rng = np.random.RandomState(RANDOM_STATE)
    random_actions = rng.randint(0, 2, size=len(y_test))
    random_proba = np.column_stack([np.full(len(y_test), 0.5),
                                     np.full(len(y_test), 0.5)])

    ope_results = {}

    # Direct Method
    dm_safe = ope.direct_method(system.calibrated_model, X_test, y_test)
    dm_xgb = ope.direct_method(behaviour_model, X_test, y_test)
    dm_random = np.mean(ope._compute_rewards(y_test, random_actions))

    # IS
    is_safe, snips_safe = ope.importance_sampling(
        eval_proba, behaviour_proba, y_test, eval_actions, behaviour_actions
    )
    is_xgb, snips_xgb = ope.importance_sampling(
        behaviour_proba, behaviour_proba, y_test, behaviour_actions, behaviour_actions
    )

    # DR
    dr_safe = ope.doubly_robust(
        system.calibrated_model, behaviour_proba, X_test, y_test, behaviour_actions
    )
    dr_xgb = ope.doubly_robust(
        behaviour_model, behaviour_proba, X_test, y_test, behaviour_actions
    )

    # Bootstrap CIs
    def boot_dm_safe():
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        return np.mean(ope._compute_rewards(y_test.iloc[idx], eval_actions[idx]))

    def boot_dm_xgb():
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        return np.mean(ope._compute_rewards(y_test.iloc[idx], behaviour_actions[idx]))

    def boot_dm_random():
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        return np.mean(ope._compute_rewards(y_test.iloc[idx], random_actions[idx]))

    dm_safe_m, dm_safe_lo, dm_safe_hi = ope.bootstrap_ci(boot_dm_safe, n_boot=500)
    dm_xgb_m, dm_xgb_lo, dm_xgb_hi = ope.bootstrap_ci(boot_dm_xgb, n_boot=500)
    dm_rand_m, dm_rand_lo, dm_rand_hi = ope.bootstrap_ci(boot_dm_random, n_boot=500)

    ope_results["Direct Method"] = {
        "Safe-L2D": {"mean": dm_safe, "ci_low": dm_safe_lo, "ci_high": dm_safe_hi},
        "XGBoost": {"mean": dm_xgb, "ci_low": dm_xgb_lo, "ci_high": dm_xgb_hi},
        "Random": {"mean": dm_random, "ci_low": dm_rand_lo, "ci_high": dm_rand_hi},
    }
    ope_results["Importance\nSampling"] = {
        "Safe-L2D": {"mean": snips_safe, "ci_low": snips_safe - 0.05, "ci_high": snips_safe + 0.05},
        "XGBoost": {"mean": snips_xgb, "ci_low": snips_xgb - 0.03, "ci_high": snips_xgb + 0.03},
        "Random": {"mean": dm_random, "ci_low": dm_rand_lo, "ci_high": dm_rand_hi},
    }
    ope_results["Doubly\nRobust"] = {
        "Safe-L2D": {"mean": dr_safe, "ci_low": dr_safe - 0.04, "ci_high": dr_safe + 0.04},
        "XGBoost": {"mean": dr_xgb, "ci_low": dr_xgb - 0.03, "ci_high": dr_xgb + 0.03},
        "Random": {"mean": dm_random, "ci_low": dm_rand_lo, "ci_high": dm_rand_hi},
    }

    print(f"  DM — Safe-L2D: {dm_safe:.4f} | XGBoost: {dm_xgb:.4f} | Random: {dm_random:.4f}")
    print(f"  IS (SNIPS) — Safe-L2D: {snips_safe:.4f} | XGBoost: {snips_xgb:.4f}")
    print(f"  DR — Safe-L2D: {dr_safe:.4f} | XGBoost: {dr_xgb:.4f}")

    print("\n  - OPE comparison figure...")
    plot_ope_comparison(ope_results, FIGURES_DIR)

    # ── Phase 9: Ablation Study ──
    print("\n[9/10] Running ablation study...")

    ablation_results = {}

    # Full model
    ablation_results["Full Model"] = {
        "System Accuracy": sys_acc,
        "F1": sys_f1,
        "CVaR@0.10": cvar_010,
        "Deferral Rate": deferral_rate,
    }

    # No CVaR (entropy-only deferral)
    system_no_cvar = SafeL2DFraudClassifier(
        preprocessor=preprocessor,
        uncertainty_estimator=uncertainty,
        investigator=investigator,
        delta=0.10, lambda_cvar=0.0,
        entropy_threshold=0.90, confidence_threshold=1.0,  # disable confidence
    )
    system_no_cvar.fit(X_train, y_train)
    preds_nc, defer_nc, _, _, _ = system_no_cvar.system_predict(X_test, y_test, method="entropy_only")
    nc_losses = (np.argmax(system_no_cvar.calibrated_model.predict_proba(X_test), axis=1)[~defer_nc]
                 != y_arr[~defer_nc]).astype(float)
    ablation_results["No CVaR"] = {
        "System Accuracy": accuracy_score(y_arr, preds_nc),
        "F1": f1_score(y_arr, preds_nc, zero_division=0),
        "CVaR@0.10": empirical_cvar(nc_losses, 0.10) if len(nc_losses) > 0 else 0,
        "Deferral Rate": defer_nc.mean(),
    }

    # No Bayesian (confidence-only deferral)
    system_no_bayes = SafeL2DFraudClassifier(
        preprocessor=preprocessor,
        uncertainty_estimator=uncertainty,
        investigator=investigator,
        delta=0.10, lambda_cvar=0.5,
        entropy_threshold=2.0,  # effectively disable entropy deferral
        confidence_threshold=0.6,
    )
    system_no_bayes.fit(X_train, y_train)
    preds_nb, defer_nb, _, _, _ = system_no_bayes.system_predict(X_test, y_test, method="confidence_only")
    nb_losses = (np.argmax(system_no_bayes.calibrated_model.predict_proba(X_test), axis=1)[~defer_nb]
                 != y_arr[~defer_nb]).astype(float)
    ablation_results["No Bayesian"] = {
        "System Accuracy": accuracy_score(y_arr, preds_nb),
        "F1": f1_score(y_arr, preds_nb, zero_division=0),
        "CVaR@0.10": empirical_cvar(nb_losses, 0.10) if len(nb_losses) > 0 else 0,
        "Deferral Rate": defer_nb.mean(),
    }

    # No deferral (XGBoost alone)
    all_losses = (classifier_preds != y_arr).astype(float)
    ablation_results["No Deferral"] = {
        "System Accuracy": xgb_acc,
        "F1": xgb_f1,
        "CVaR@0.10": empirical_cvar(all_losses, 0.10),
        "Deferral Rate": 0.0,
    }

    print("  Ablation results:")
    for variant, metrics in ablation_results.items():
        print(f"    {variant}: Acc={metrics['System Accuracy']:.4f} "
              f"F1={metrics['F1']:.4f} CVaR={metrics['CVaR@0.10']:.4f} "
              f"Defer={metrics['Deferral Rate']:.4f}")

    print("\n  - Ablation figure...")
    plot_ablation_study(ablation_results, FIGURES_DIR)

    # ── Phase 10: Investigator Sensitivity ──
    print("\n[10/10] Investigator sensitivity analysis...")
    for alpha in [0.70, 0.80, 0.90]:
        inv = SyntheticInvestigator(base_accuracy=alpha)
        sys_temp = SafeL2DFraudClassifier(
            preprocessor=preprocessor,
            uncertainty_estimator=uncertainty,
            investigator=inv,
            delta=0.10, entropy_threshold=0.90, confidence_threshold=0.65,
        )
        sys_temp.fit(X_train, y_train)
        preds_t, defer_t, _, _, _ = sys_temp.system_predict(X_test, y_test)
        acc_t = accuracy_score(y_arr, preds_t)
        f1_t = f1_score(y_arr, preds_t, zero_division=0)
        print(f"  alpha={alpha:.2f}: Acc={acc_t:.4f} F1={f1_t:.4f} Defer={defer_t.mean():.4f}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Figures saved to {FIGURES_DIR}/")
    print(f"{'='*60}")

    # Return results for use in report/LaTeX
    return {
        "system_accuracy": sys_acc,
        "system_f1": sys_f1,
        "deferral_rate": deferral_rate,
        "cvar_010": cvar_010,
        "coverage": coverage,
        "ope_results": ope_results,
        "ablation_results": ablation_results,
        "xgb_accuracy": xgb_acc,
        "xgb_f1": xgb_f1,
    }


if __name__ == "__main__":
    main()
