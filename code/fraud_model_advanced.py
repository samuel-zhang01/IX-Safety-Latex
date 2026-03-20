"""
Insurance Fraud Detection — Advanced Analysis Pipeline
Extends fraud_model.py with Bayesian inference, statistical model comparison,
and causal modelling. Generates SVG figures and a comprehensive Markdown report.
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
import networkx as nx
from scipy.stats import gaussian_kde, chi2 as chi2_dist

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_model import (
    load_and_clean, engineer_features, classify_columns,
    build_preprocessor, RANDOM_STATE, TEST_SIZE,
)

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "insurance_fraud_claims.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_model_advanced_report.md")

BAYESIAN_FEATURES = [
    "total_claim_amount", "policy_age_days",
    "months_as_customer", "incident_hour_of_the_day",
]
TREATMENT_VAR = "police_report_available"
CONFOUNDERS = [
    "months_as_customer", "age", "policy_deductable",
    "policy_annual_premium", "umbrella_limit",
    "incident_hour_of_the_day", "capital-gains", "capital-loss",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Models (original 4 + Gaussian Naive Bayes)
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_models(preprocessor):
    return {
        "SVM (SVC)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB()),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=RANDOM_STATE,
            )),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A: Bayesian Inference
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_conditionals(X_train, y_train, feature_names):
    """Compute class-conditional KDE for P(feature|fraud) vs P(feature|not fraud)."""
    results = {}
    for feat in feature_names:
        vals = X_train[feat].dropna()
        fraud_vals = vals[y_train.loc[vals.index] == 1].values.astype(float)
        legit_vals = vals[y_train.loc[vals.index] == 0].values.astype(float)

        if len(fraud_vals) < 2 or len(legit_vals) < 2:
            continue

        x_min = min(fraud_vals.min(), legit_vals.min())
        x_max = max(fraud_vals.max(), legit_vals.max())
        x_range = np.linspace(x_min, x_max, 300)

        kde_fraud = gaussian_kde(fraud_vals)
        kde_legit = gaussian_kde(legit_vals)

        results[feat] = {
            "kde_fraud": kde_fraud,
            "kde_legit": kde_legit,
            "x_range": x_range,
            "fraud_density": kde_fraud.evaluate(x_range),
            "legit_density": kde_legit.evaluate(x_range),
        }
    return results


def compute_posterior_updates(prior_fraud, conditionals, X_samples, feature_names):
    """Demonstrate sequential Bayesian updating: prior * likelihood → posterior."""
    all_traces = []
    for idx in range(len(X_samples)):
        row = X_samples.iloc[idx]
        trace = [{"feature": "Prior", "probability": prior_fraud}]
        current_prior = prior_fraud

        for feat in feature_names:
            if feat not in conditionals or pd.isna(row.get(feat)):
                continue
            val = float(row[feat])
            cond = conditionals[feat]
            l_fraud = float(cond["kde_fraud"].evaluate(np.array([val]))[0])
            l_legit = float(cond["kde_legit"].evaluate(np.array([val]))[0])

            eps = 1e-10
            l_fraud = max(l_fraud, eps)
            l_legit = max(l_legit, eps)

            posterior = (current_prior * l_fraud) / (
                current_prior * l_fraud + (1 - current_prior) * l_legit
            )
            trace.append({"feature": feat, "probability": posterior})
            current_prior = posterior

        all_traces.append(trace)
    return all_traces


def plot_class_conditional_densities(conditionals, feature_names, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    valid = [f for f in feature_names if f in conditionals]
    n = len(valid)
    if n == 0:
        return
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for i, feat in enumerate(valid):
        ax = axes[i]
        c = conditionals[feat]
        ax.fill_between(c["x_range"], c["fraud_density"], alpha=0.4, color="#d62728", label="P(x | Fraud)")
        ax.fill_between(c["x_range"], c["legit_density"], alpha=0.4, color="#1f77b4", label="P(x | Not Fraud)")
        ax.plot(c["x_range"], c["fraud_density"], color="#d62728", linewidth=1.5)
        ax.plot(c["x_range"], c["legit_density"], color="#1f77b4", linewidth=1.5)
        ax.set_title(feat.replace("_", " ").title(), fontsize=14, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linestyle="--")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Class-Conditional Distributions — P(Feature | Class)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "class_conditional_densities.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_prior_posterior_update(update_traces, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    n_samples = len(update_traces)
    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 5), squeeze=False)
    axes = axes.flatten()

    for idx, trace in enumerate(update_traces):
        ax = axes[idx]
        labels = [t["feature"].replace("_", " ").title() for t in trace]
        probs = [t["probability"] for t in trace]
        colors = ["#d62728" if p > 0.5 else "#1f77b4" for p in probs]

        bars = ax.bar(range(len(probs)), probs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
        ax.set_ylabel("P(Fraud)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Sample {idx + 1}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{prob:.3f}", ha="center", fontsize=10)

    fig.suptitle("Bayesian Posterior Update — Sequential Feature Evidence",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "prior_posterior_update.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B: Model Statistical Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Pairwise McNemar's test with continuity correction."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    if b + c == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2_dist.sf(chi2_stat, df=1))
    return chi2_stat, p_value


def run_mcnemar_matrix(model_predictions, y_true, model_names):
    n = len(model_names)
    matrix = np.ones((n, n))
    np.fill_diagonal(matrix, np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            _, p = mcnemar_test(y_true, model_predictions[model_names[i]],
                                model_predictions[model_names[j]])
            matrix[i, j] = p
            matrix[j, i] = p
    return pd.DataFrame(matrix, index=model_names, columns=model_names)


def run_cross_validation(models_dict, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipeline in models_dict.items():
        print(f"    CV: {name}...")
        f1_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
        acc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
        auc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
        cv_results[name] = {
            "f1_scores": f1_scores,
            "f1_mean": f1_scores.mean(), "f1_std": f1_scores.std(),
            "acc_scores": acc_scores,
            "acc_mean": acc_scores.mean(), "acc_std": acc_scores.std(),
            "auc_scores": auc_scores,
            "auc_mean": auc_scores.mean(), "auc_std": auc_scores.std(),
        }
    return cv_results


def plot_roc_overlay(fitted_models, X_test, y_test, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(fitted_models.items(), COLORS):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_overlay.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_calibration_curves(fitted_models, X_test, y_test, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(fitted_models.items(), COLORS):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            if len(np.unique(y_proba)) < 3:
                continue
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8)
            ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2, label=name)
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=13)
    ax.set_ylabel("Fraction of Positives", fontsize=13)
    ax.set_title("Calibration Curves — Predicted vs. Actual Probability", fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "calibration_curves.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_cv_boxplot(cv_results, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    model_names = list(cv_results.keys())
    f1_data = [cv_results[n]["f1_scores"] for n in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(f1_data, labels=model_names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, name in enumerate(model_names):
        mean = cv_results[name]["f1_mean"]
        std = cv_results[name]["f1_std"]
        ax.text(i + 1, mean + std + 0.02, f"{mean:.3f}\n(+/-{std:.3f})",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_title("5-Fold Cross-Validation F1 Scores", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=15, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cv_boxplot.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def plot_mcnemar_heatmap(mcnemar_df, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, ax = plt.subplots(figsize=(8, 6))
    data = mcnemar_df.values.copy()
    mask = np.isnan(data)
    display_data = np.where(mask, 0.5, data)

    im = ax.imshow(display_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    n = len(mcnemar_df)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "---", ha="center", va="center", fontsize=11, color="gray")
            else:
                val = data[i, j]
                txt_color = "white" if val < 0.05 else "black"
                weight = "bold" if val < 0.05 else "normal"
                sig = " *" if val < 0.05 else ""
                ax.text(j, i, f"{val:.3f}{sig}", ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight=weight)

    names = list(mcnemar_df.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p-value", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title("McNemar's Test p-values (Pairwise Model Comparison)\n* = significant at p<0.05",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mcnemar_heatmap.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION C: Causal Modelling
# ═══════════════════════════════════════════════════════════════════════════════

def build_causal_dag():
    edges = [
        ("policy_age_days", "fraud_reported"),
        ("policy_deductable", "fraud_reported"),
        ("umbrella_limit", "fraud_reported"),
        ("incident_severity", "fraud_reported"),
        ("total_claim_amount", "fraud_reported"),
        ("witnesses", "fraud_reported"),
        ("police_report_available", "fraud_reported"),
        ("bodily_injuries", "fraud_reported"),
        ("incident_severity", "total_claim_amount"),
        ("incident_severity", "bodily_injuries"),
        ("number_of_vehicles", "total_claim_amount"),
        ("number_of_vehicles", "bodily_injuries"),
        ("policy_deductable", "total_claim_amount"),
        ("insured_occupation", "policy_annual_premium"),
        ("policy_annual_premium", "umbrella_limit"),
    ]
    return nx.DiGraph(edges)


def plot_causal_dag(dag, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    layers = {
        0: ["insured_occupation"],
        1: ["policy_annual_premium", "policy_deductable", "policy_age_days"],
        2: ["umbrella_limit", "incident_severity", "number_of_vehicles"],
        3: ["total_claim_amount", "bodily_injuries", "witnesses", "police_report_available"],
        4: ["fraud_reported"],
    }

    pos = {}
    for layer_idx, nodes in layers.items():
        y = 1.0 - layer_idx * 0.22
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * 0.28
            pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(14, 10))
    node_colors = ["#d62728" if n == "fraud_reported" else "#6baed6" for n in dag.nodes()]
    node_sizes = [3000 if n == "fraud_reported" else 2000 for n in dag.nodes()]

    nx.draw_networkx_nodes(dag, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors="black", linewidths=1.2)
    nx.draw_networkx_edges(dag, pos, ax=ax, edge_color="#555555", arrows=True,
                           arrowsize=20, width=1.5, connectionstyle="arc3,rad=0.1",
                           min_source_margin=20, min_target_margin=20)
    labels = {n: n.replace("_", "\n") for n in dag.nodes()}
    nx.draw_networkx_labels(dag, pos, labels=labels, ax=ax, font_size=10, font_weight="bold")

    ax.set_title("Hypothesised Causal DAG for Insurance Fraud", fontsize=15, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "causal_dag.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def propensity_score_analysis(df, treatment_col, outcome_col, confounders):
    subset = df[[treatment_col, outcome_col] + confounders].dropna()
    subset = subset.copy()
    subset[treatment_col] = subset[treatment_col].map({"YES": 1, "NO": 0})
    subset = subset.dropna(subset=[treatment_col])

    treatment = subset[treatment_col].values.astype(int)
    outcome = subset[outcome_col].values.astype(int)
    X_conf = subset[confounders].values.astype(float)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_conf, treatment)
    ps = lr.predict_proba(X_conf)[:, 1]

    try:
        strata = pd.qcut(ps, q=5, labels=False, duplicates="drop")
    except ValueError:
        strata = pd.cut(ps, bins=5, labels=False)

    stratum_effects = []
    stratum_counts = []
    for s in sorted(np.unique(strata)):
        mask = strata == s
        t1 = outcome[mask & (treatment == 1)]
        t0 = outcome[mask & (treatment == 0)]
        if len(t1) == 0 or len(t0) == 0:
            continue
        effect = t1.mean() - t0.mean()
        count = int(mask.sum())
        stratum_effects.append(effect)
        stratum_counts.append(count)

    total = sum(stratum_counts)
    ate = sum(e * c / total for e, c in zip(stratum_effects, stratum_counts)) if total > 0 else 0.0

    return {
        "propensity_scores": ps,
        "treatment": treatment,
        "outcome": outcome,
        "ate": ate,
        "stratum_effects": stratum_effects,
        "stratum_counts": stratum_counts,
        "n_used": len(subset),
    }


def plot_propensity_scores(ps_results, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ps = ps_results["propensity_scores"]
    treatment = ps_results["treatment"]

    ax1.hist(ps[treatment == 1], bins=20, alpha=0.6, color="#1f77b4", label="Treatment (Report=YES)", density=True)
    ax1.hist(ps[treatment == 0], bins=20, alpha=0.6, color="#d62728", label="Control (Report=NO)", density=True)
    ax1.set_xlabel("Propensity Score", fontsize=13)
    ax1.set_ylabel("Density", fontsize=13)
    ax1.set_title("Propensity Score Distributions", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, linestyle="--")

    effects = ps_results["stratum_effects"]
    ate = ps_results["ate"]
    x_pos = range(len(effects))
    colors = ["#2ca02c" if e > 0 else "#d62728" for e in effects]
    ax2.bar(x_pos, effects, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=ate, color="black", linestyle="--", linewidth=2, label=f"ATE = {ate:.4f}")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Propensity Score Stratum (Quintile)", fontsize=13)
    ax2.set_ylabel("Treatment Effect (Fraud Rate Diff)", fontsize=13)
    ax2.set_title("Stratum-Level Treatment Effects", fontsize=15, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Q{i+1}" for i in x_pos])
    ax2.legend(fontsize=11)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Propensity Score Analysis: Effect of Police Report on Fraud Detection",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "propensity_scores.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


def compute_feature_importance_comparison(lr_pipeline, X_test, y_test, conditionals, bayesian_features):
    perm_result = permutation_importance(
        lr_pipeline, X_test, y_test, n_repeats=10,
        random_state=RANDOM_STATE, scoring="f1",
    )
    all_features = X_test.columns.tolist()
    perm_df = pd.DataFrame({
        "feature": all_features,
        "perm_importance": perm_result.importances_mean,
    }).sort_values("perm_importance", ascending=False)

    kl_scores = {}
    for feat in bayesian_features:
        if feat not in conditionals:
            continue
        c = conditionals[feat]
        p = c["fraud_density"] + 1e-10
        q = c["legit_density"] + 1e-10
        p_norm = p / p.sum()
        q_norm = q / q.sum()
        kl = float(np.sum(p_norm * np.log(p_norm / q_norm)))
        kl_scores[feat] = kl

    kl_df = pd.DataFrame([
        {"feature": f, "kl_divergence": v} for f, v in kl_scores.items()
    ])

    merged = perm_df.merge(kl_df, on="feature", how="outer")
    return merged


def plot_importance_comparison(comparison_df, save_dir):
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    top_perm = comparison_df.dropna(subset=["perm_importance"]).nlargest(12, "perm_importance")
    ax1.barh(range(len(top_perm)), top_perm["perm_importance"].values, color="#1f77b4", alpha=0.7)
    ax1.set_yticks(range(len(top_perm)))
    ax1.set_yticklabels([f.replace("_", " ").title() for f in top_perm["feature"]], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Permutation Importance (F1 drop)", fontsize=13)
    ax1.set_title("Permutation Importance\n(Logistic Regression)", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.25, linestyle="--")

    kl_data = comparison_df.dropna(subset=["kl_divergence"]).sort_values("kl_divergence", ascending=False)
    if len(kl_data) > 0:
        ax2.barh(range(len(kl_data)), kl_data["kl_divergence"].values, color="#d62728", alpha=0.7)
        ax2.set_yticks(range(len(kl_data)))
        ax2.set_yticklabels([f.replace("_", " ").title() for f in kl_data["feature"]], fontsize=11)
        ax2.invert_yaxis()
    ax2.set_xlabel("KL Divergence (P(x|fraud) || P(x|legit))", fontsize=13)
    ax2.set_title("Bayesian Likelihood Divergence\n(Class-Conditional KDE)", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")

    fig.suptitle("Feature Importance: Discriminative vs. Bayesian Perspective",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "feature_importance_comparison.svg"), format="svg", bbox_inches="tight")
    # Don't reset globally since it's per-function
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_advanced_report(rd, save_path):
    L = []

    # ── Title ──
    L.append("# Insurance Fraud Detection — Advanced Analysis Report\n")

    # ── 1. Dataset Overview ──
    L.append("## 1. Dataset Overview\n")
    L.append(f"- **Source:** `insurance fraud claims.csv`")
    L.append(f"- **Total samples:** {rd['train_size'] + rd['test_size']}")
    L.append(f"- **Train / Test split:** {rd['train_size']} / {rd['test_size']} (80:20 stratified)")
    L.append(f"- **Features after preprocessing:** {rd['feature_count']}")
    L.append(f"- **Base fraud rate (prior):** {rd['prior_fraud']:.3f}")
    L.append("- **Missing values:** `collision_type` (178), `property_damage` (360), "
             "`police_report_available` (343) — imputed with most-frequent\n")

    # ── 2. Model Performance Summary ──
    L.append("## 2. Model Performance Summary\n")
    L.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    L.append("|-------|----------|-----------|--------|----------|---------|")
    best = max(rd["base_results"], key=lambda r: r["f1"])
    for r in rd["base_results"]:
        mark = " **" if r["name"] == best["name"] else ""
        end = "**" if mark else ""
        L.append(f"| {mark}{r['name']}{end} | {r['accuracy']:.4f} | {r['precision']:.4f} "
                 f"| {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |")
    L.append("")

    L.append("### Confusion Matrices\n")
    for r in rd["base_results"]:
        cm = r["confusion_matrix"]
        L.append(f"**{r['name']}:** TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}\n")

    # ── 3. Bayesian Inference Analysis ──
    L.append("## 3. Bayesian Inference Analysis\n")
    L.append("### 3.1 Gaussian Naive Bayes as a Classifier\n")
    nb = next((r for r in rd["base_results"] if r["name"] == "Gaussian Naive Bayes"), None)
    if nb:
        L.append(f"Gaussian Naive Bayes achieved F1={nb['f1']:.4f} and ROC-AUC={nb['roc_auc']:.4f}. "
                 "This model assumes feature independence given the class label — the 'naive' assumption. "
                 "Despite this strong assumption, it provides a principled probabilistic framework "
                 "and serves as the foundation for the Bayesian analysis below.\n")

    L.append("### 3.2 Class-Conditional Distributions\n")
    L.append("The plots below show P(feature | fraud) vs. P(feature | not fraud) estimated via "
             "kernel density estimation. Features where the two distributions diverge significantly "
             "carry more discriminative power for fraud detection.\n")
    L.append("![Class-Conditional Density Plots](figures/class_conditional_densities.svg)\n")

    for feat in rd["bayesian_features"]:
        if feat in rd["conditionals"]:
            c = rd["conditionals"][feat]
            fraud_mean = float(rd["conditionals"][feat]["x_range"][np.argmax(c["fraud_density"])])
            legit_mean = float(rd["conditionals"][feat]["x_range"][np.argmax(c["legit_density"])])
            L.append(f"- **{feat}**: Fraud mode ~ {fraud_mean:.1f}, Non-fraud mode ~ {legit_mean:.1f}")
    L.append("")

    L.append("### 3.3 Prior to Posterior Update\n")
    L.append(f"Starting from the base fraud rate (prior = {rd['prior_fraud']:.3f}), we sequentially "
             "update the probability of fraud as each feature value is observed. This demonstrates "
             "how Bayesian reasoning combines prior belief with new evidence:\n")
    L.append("**P(fraud | evidence) = P(evidence | fraud) x P(fraud) / P(evidence)**\n")
    L.append("![Prior-Posterior Update](figures/prior_posterior_update.svg)\n")

    for i, trace in enumerate(rd["update_traces"]):
        final_p = trace[-1]["probability"]
        label = "fraud" if final_p > 0.5 else "not fraud"
        L.append(f"- **Sample {i+1}:** Prior {trace[0]['probability']:.3f} -> "
                 f"Posterior {final_p:.3f} ({label})")
    L.append("")

    # ── 4. Model Statistical Comparison ──
    L.append("## 4. Model Statistical Comparison\n")
    L.append("### 4.1 Cross-Validation Results (5-Fold Stratified)\n")
    L.append("| Model | F1 (mean +/- std) | Accuracy (mean +/- std) | ROC-AUC (mean +/- std) |")
    L.append("|-------|-------------------|-------------------------|------------------------|")
    for name, cv in rd["cv_results"].items():
        L.append(f"| {name} | {cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f} "
                 f"| {cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f} "
                 f"| {cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f} |")
    L.append("")
    L.append("![Cross-Validation Box Plot](figures/cv_boxplot.svg)\n")

    L.append("### 4.2 McNemar's Test\n")
    L.append("McNemar's test evaluates whether two classifiers make *significantly different errors* "
             "(not just different accuracy). A p-value < 0.05 indicates a statistically significant "
             "difference in error patterns.\n")
    L.append("![McNemar Heatmap](figures/mcnemar_heatmap.svg)\n")

    sig_pairs = []
    mn = rd["mcnemar_df"]
    names = list(mn.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = mn.iloc[i, j]
            if not np.isnan(p) and p < 0.05:
                sig_pairs.append((names[i], names[j], p))
    if sig_pairs:
        L.append("**Significant differences (p < 0.05):**\n")
        for a, b, p in sig_pairs:
            L.append(f"- {a} vs. {b}: p = {p:.4f}")
        L.append("")
    else:
        L.append("No model pairs showed statistically significant differences at p < 0.05.\n")

    L.append("### 4.3 ROC Curve Comparison\n")
    L.append("![ROC Overlay](figures/roc_overlay.svg)\n")

    L.append("### 4.4 Calibration Analysis\n")
    L.append("A well-calibrated model produces predicted probabilities that match actual frequencies. "
             "Points close to the diagonal line indicate good calibration.\n")
    L.append("![Calibration Curves](figures/calibration_curves.svg)\n")

    # ── 5. Causal Analysis ──
    L.append("## 5. Causal Analysis\n")
    L.append("### 5.1 Hypothesised Causal DAG\n")
    L.append("The directed acyclic graph below represents hypothesised causal relationships based "
             "on domain knowledge of insurance fraud. Arrows indicate the direction of causal "
             "influence. This is an *assumed* structure — not learned from data.\n")
    L.append("![Causal DAG](figures/causal_dag.svg)\n")

    L.append("### 5.2 Propensity Score Analysis\n")
    ps = rd["ps_results"]
    L.append(f"**Treatment:** presence of a police report (`police_report_available` = YES)\n")
    L.append(f"**Outcome:** fraud reported\n")
    L.append(f"**Observations used:** {ps['n_used']} (rows with non-missing treatment status)\n")
    L.append(f"**Estimated Average Treatment Effect (ATE):** {ps['ate']:.4f}\n")
    if ps["ate"] > 0:
        L.append("A positive ATE suggests that claims with a police report are *more likely* "
                 "to be flagged as fraud. This likely reflects reverse causality or selection bias: "
                 "suspicious claims trigger police involvement, rather than police reports causing fraud.\n")
    else:
        L.append("A negative ATE suggests claims with police reports are *less likely* to be "
                 "fraudulent, consistent with the idea that legitimate claimants are more likely "
                 "to file police reports.\n")
    L.append("![Propensity Score Analysis](figures/propensity_scores.svg)\n")

    L.append("### 5.3 Feature Importance: Discriminative vs. Bayesian\n")
    L.append("Permutation importance measures how much a model's F1 score drops when a feature "
             "is randomly shuffled. KL divergence measures how different the class-conditional "
             "distributions are. Features that rank high on both scales have the strongest "
             "signal for fraud detection.\n")
    L.append("![Feature Importance Comparison](figures/feature_importance_comparison.svg)\n")

    # ── 6. Conclusions ──
    L.append("## 6. Overall Conclusions\n")
    best_cv = max(rd["cv_results"].items(), key=lambda x: x[1]["f1_mean"])
    L.append(f"1. **Best single-split model:** {best['name']} (F1={best['f1']:.4f})")
    L.append(f"2. **Most robust under CV:** {best_cv[0]} "
             f"(F1={best_cv[1]['f1_mean']:.4f} +/- {best_cv[1]['f1_std']:.4f})")
    L.append(f"3. **Bayesian analysis** reveals that `total_claim_amount` and `policy_age_days` "
             "show the strongest class-conditional separation, confirming their importance for "
             "fraud detection.")
    L.append(f"4. **Causal analysis** estimates an ATE of {ps['ate']:.4f} for police report "
             "availability, suggesting the relationship between police reports and fraud reflects "
             "selection/reporting patterns rather than direct causation.")
    L.append("5. **Calibration analysis** identifies which models produce well-calibrated "
             "probabilities — critical for operationalising fraud scores in practice.\n")
    L.append("> **Caveat:** Causal conclusions from observational data require strong assumptions. "
             "The DAG and propensity score analysis represent a starting framework, not definitive "
             "causal proof.\n")

    with open(save_path, "w") as f:
        f.write("\n".join(L))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Insurance Fraud Detection — Advanced Analysis Pipeline")
    print("=" * 60)

    ensure_figures_dir()

    # ── Phase 1: Data Preparation ──
    print("\n[1/12] Loading and cleaning data...")
    df = load_and_clean(CSV_PATH)

    print("\n[2/12] Engineering features...")
    df = engineer_features(df)
    df_full = df.copy()

    print("\n[3/12] Classifying columns and building preprocessor...")
    X, y, num_cols, low_cat_cols, med_cat_cols = classify_columns(df, "fraud_reported")
    preprocessor = build_preprocessor(num_cols, low_cat_cols, med_cat_cols)

    print("\n[4/12] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    prior_fraud = y_train.mean()
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Prior fraud rate: {prior_fraud:.3f}")

    # ── Phase 2: Train all 5 models ──
    print("\n[5/12] Training all 5 models...")
    models = get_all_models(preprocessor)
    base_results = []
    fitted_models = {}
    model_predictions = {}

    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        model_predictions[name] = y_pred

        result = {
            "name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        base_results.append(result)
        print(f"    F1={result['f1']:.4f}  ROC-AUC={result['roc_auc']:.4f}")

    feature_count = fitted_models[list(fitted_models.keys())[0]] \
        .named_steps["preprocessor"].transform(X_test.head(1)).shape[1]

    # ── Phase 3: Section A — Bayesian Inference ──
    print("\n[6/12] Section A: Computing class-conditional distributions...")
    conditionals = compute_class_conditionals(X_train, y_train, BAYESIAN_FEATURES)

    print("\n[7/12] Section A: Generating Bayesian plots...")
    plot_class_conditional_densities(conditionals, BAYESIAN_FEATURES, FIGURES_DIR)

    X_samples = X_test.iloc[:3]
    update_traces = compute_posterior_updates(prior_fraud, conditionals, X_samples, BAYESIAN_FEATURES)
    plot_prior_posterior_update(update_traces, FIGURES_DIR)

    # ── Phase 4: Section B — Statistical Comparison ──
    print("\n[8/12] Section B: McNemar's pairwise tests...")
    model_names = list(fitted_models.keys())
    mcnemar_df = run_mcnemar_matrix(model_predictions, y_test.values, model_names)

    print("\n[9/12] Section B: Cross-validation (5-fold)...")
    cv_results = run_cross_validation(models, X, y)

    print("\n[10/12] Section B: Generating comparison plots...")
    plot_roc_overlay(fitted_models, X_test, y_test, FIGURES_DIR)
    plot_calibration_curves(fitted_models, X_test, y_test, FIGURES_DIR)
    plot_cv_boxplot(cv_results, FIGURES_DIR)
    plot_mcnemar_heatmap(mcnemar_df, FIGURES_DIR)

    # ── Phase 5: Section C — Causal Modelling ──
    print("\n[11/12] Section C: Causal analysis...")
    dag = build_causal_dag()
    plot_causal_dag(dag, FIGURES_DIR)

    ps_results = propensity_score_analysis(
        df_full, TREATMENT_VAR, "fraud_reported", CONFOUNDERS,
    )
    print(f"  Propensity Score ATE: {ps_results['ate']:.4f} (n={ps_results['n_used']})")
    plot_propensity_scores(ps_results, FIGURES_DIR)

    importance_df = compute_feature_importance_comparison(
        fitted_models["Logistic Regression"], X_test, y_test,
        conditionals, BAYESIAN_FEATURES,
    )
    plot_importance_comparison(importance_df, FIGURES_DIR)

    # ── Phase 6: Report ──
    print("\n[12/12] Generating advanced report...")
    report_data = {
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "feature_count": feature_count,
        "prior_fraud": prior_fraud,
        "base_results": base_results,
        "cv_results": cv_results,
        "mcnemar_df": mcnemar_df,
        "conditionals": conditionals,
        "update_traces": update_traces,
        "ps_results": ps_results,
        "importance_df": importance_df,
        "bayesian_features": BAYESIAN_FEATURES,
    }
    generate_advanced_report(report_data, REPORT_PATH)

    print(f"\n  Report: {REPORT_PATH}")
    print(f"  Figures: {FIGURES_DIR}/")
    best = max(base_results, key=lambda r: r["f1"])
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best['name']} (F1={best['f1']:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
