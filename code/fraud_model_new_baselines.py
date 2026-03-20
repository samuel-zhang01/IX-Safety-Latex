"""
Insurance Fraud Detection — New Baselines for Safe-L2D-Fraud Comparison
Implements:
  1. Mozannar-Sontag End-to-End L2D Baseline (3-class surrogate)
  2. Cost-Sensitive XGBoost Baseline (no deferral)
  3. Full comparison table with bootstrap 95% CIs (B=2000)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_model import (
    load_and_clean, engineer_features, classify_columns,
    build_preprocessor, RANDOM_STATE, TEST_SIZE,
)
from fraud_model_advanced import compute_class_conditionals, BAYESIAN_FEATURES
from fraud_model_l2d import (
    BayesianUncertaintyEstimator, SyntheticInvestigator,
    SafeL2DFraudClassifier, empirical_cvar, confidence_baseline_curve,
)

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "insurance_fraud_claims.csv")

# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 1: Mozannar-Sontag End-to-End L2D
# ═══════════════════════════════════════════════════════════════════════════════

class MozannarSontagL2D:
    """
    End-to-end Learning to Defer following Mozannar & Sontag (ICML 2020).

    Trains a 3-class model:
      class 0 = predict not-fraud
      class 1 = predict fraud
      class 2 = defer to expert

    Training labels are assigned via the surrogate cost approach:
      For each sample (x, y), compute costs of each action:
        c_0 = 1[y=1]   (cost of predicting not-fraud when it IS fraud)
        c_1 = 1[y=0]   (cost of predicting fraud when it is NOT fraud)
        c_perp = 1 - alpha  (expected expert error rate)
      Label = argmin(c_0, c_1, c_perp)

    For ambiguous samples where the base classifier is likely wrong,
    deferral becomes the lowest-cost action.
    """

    def __init__(self, preprocessor, expert_accuracy=0.85, random_state=RANDOM_STATE):
        self.preprocessor = preprocessor
        self.alpha = expert_accuracy
        self.random_state = random_state
        self.model = None
        self.base_clf = None

    def _assign_surrogate_labels(self, X_train, y_train):
        """
        Assign 3-class training labels using the Mozannar-Sontag cost structure.

        Step 1: Train a preliminary classifier to estimate per-sample difficulty.
        Step 2: For each sample, compute costs and assign the minimum-cost action.
        """
        # Step 1: Train a preliminary XGBoost to get per-sample confidence
        self.base_clf = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=self.random_state,
            )),
        ])
        self.base_clf.fit(X_train, y_train)

        # Get calibrated probabilities via cross-val
        calibrated = CalibratedClassifierCV(self.base_clf, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_train)
        pred_class = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        y_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        n = len(y_arr)
        labels = np.zeros(n, dtype=int)

        expert_loss = 1.0 - self.alpha  # c_perp

        for i in range(n):
            # Cost of each action
            c_0 = float(y_arr[i] == 1)   # cost of predicting not-fraud
            c_1 = float(y_arr[i] == 0)   # cost of predicting fraud
            c_perp = expert_loss          # cost of deferring

            costs = [c_0, c_1, c_perp]
            best_action = int(np.argmin(costs))

            # For correctly-classified samples, the classifier action has cost 0
            # so it always wins over deferral (c_perp=0.15 > 0).
            # For misclassified samples, the classifier action has cost 1,
            # so deferral (0.15) wins.
            # However, at training time we know y, so the "correct" class always
            # has cost 0. The key insight from M&S is to also consider classifier
            # uncertainty: if the classifier is uncertain, defer.

            # Refined assignment: if the classifier would misclassify AND
            # its confidence is below a threshold, label as defer
            classifier_would_be_wrong = (pred_class[i] != y_arr[i])

            if classifier_would_be_wrong and confidence[i] < (1.0 - expert_loss):
                # Deferral is cheaper than the classifier error
                labels[i] = 2  # defer
            else:
                # The correct class action has cost 0, which is always best
                labels[i] = int(y_arr[i])  # 0 for legit, 1 for fraud

        return labels

    def fit(self, X_train, y_train):
        """Train the 3-class L2D model."""
        print("    [M-S] Assigning surrogate training labels...")
        surrogate_labels = self._assign_surrogate_labels(X_train, y_train)

        n_defer = (surrogate_labels == 2).sum()
        n_total = len(surrogate_labels)
        print(f"    [M-S] Surrogate labels: class0={np.sum(surrogate_labels==0)}, "
              f"class1={np.sum(surrogate_labels==1)}, defer={n_defer} "
              f"({100*n_defer/n_total:.1f}%)")

        # Train 3-class model
        print("    [M-S] Training 3-class XGBoost...")
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                eval_metric="mlogloss", random_state=self.random_state,
                num_class=3, objective="multi:softprob",
            )),
        ])
        self.model.fit(X_train, surrogate_labels)
        return self

    def predict(self, X):
        """
        Predict with deferral.
        Returns: (predictions, defer_mask)
          - predictions[i] in {0, 1} for non-deferred samples
          - defer_mask[i] = True if sample i is deferred
        """
        proba = self.model.predict_proba(X)
        pred_3class = np.argmax(proba, axis=1)

        defer_mask = (pred_3class == 2)
        predictions = pred_3class.copy()

        # For deferred samples, set prediction to 0 (will be overridden by expert)
        predictions[defer_mask] = 0

        return predictions, defer_mask

    def system_predict(self, X, y_true, investigator):
        """Full system: classifier + expert on deferred cases."""
        predictions, defer_mask = self.predict(X)

        if defer_mask.sum() > 0:
            X_deferred = X[defer_mask] if hasattr(X, 'loc') else X[defer_mask]
            y_deferred = y_true[defer_mask] if hasattr(y_true, 'iloc') else y_true[defer_mask]
            inv_preds = investigator.predict(X_deferred, y_deferred)
            predictions[defer_mask] = inv_preds

        return predictions, defer_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline 2: Cost-Sensitive XGBoost (No Deferral)
# ═══════════════════════════════════════════════════════════════════════════════

class CostSensitiveXGBoost:
    """
    XGBoost with scale_pos_weight to reflect asymmetric FN/FP costs.
    No deferral -- commits to every prediction.

    scale_pos_weight = 5 reflects that missing a fraud (FN) is ~5x worse
    than a false alarm (FP).
    """

    def __init__(self, preprocessor, scale_pos_weight=5.0, random_state=RANDOM_STATE):
        self.preprocessor = preprocessor
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="logloss", random_state=self.random_state,
            )),
        ])
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ═══════════════════════════════════════════════════════════════════════════════
#  Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_metrics(y_true, system_preds, defer_mask, classifier_preds,
                      delta=0.10, B=2000, seed=RANDOM_STATE):
    """
    Compute bootstrap 95% CIs for system accuracy, F1, CVaR_delta, deferral rate.

    Parameters
    ----------
    y_true : array-like, true labels
    system_preds : array-like, final system predictions (after expert handles deferred)
    defer_mask : array-like of bool, True = deferred
    classifier_preds : array-like, raw classifier predictions (before expert)
    delta : float, CVaR tail fraction
    B : int, number of bootstrap resamples
    seed : int, random seed

    Returns dict with point estimates and 95% CIs.
    """
    rng = np.random.RandomState(seed)
    y_arr = np.array(y_true)
    sp = np.array(system_preds)
    dm = np.array(defer_mask)
    cp = np.array(classifier_preds)
    n = len(y_arr)

    boot_acc = np.zeros(B)
    boot_f1 = np.zeros(B)
    boot_cvar = np.zeros(B)
    boot_defer = np.zeros(B)

    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        y_b = y_arr[idx]
        sp_b = sp[idx]
        dm_b = dm[idx]
        cp_b = cp[idx]

        boot_acc[b] = accuracy_score(y_b, sp_b)
        boot_f1[b] = f1_score(y_b, sp_b, zero_division=0)
        boot_defer[b] = dm_b.mean()

        # CVaR on non-deferred
        non_defer = ~dm_b
        if non_defer.sum() > 0:
            losses = (cp_b[non_defer] != y_b[non_defer]).astype(float)
            boot_cvar[b] = empirical_cvar(losses, delta)
        else:
            boot_cvar[b] = 0.0

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "accuracy": (accuracy_score(y_arr, sp), ci(boot_acc)),
        "f1": (f1_score(y_arr, sp, zero_division=0), ci(boot_f1)),
        "cvar_010": (empirical_cvar(
            (cp[~dm] != y_arr[~dm]).astype(float), delta) if (~dm).sum() > 0 else 0.0,
            ci(boot_cvar)),
        "deferral_rate": (dm.mean(), ci(boot_defer)),
    }


def bootstrap_metrics_no_defer(y_true, preds, delta=0.10, B=2000, seed=RANDOM_STATE):
    """Bootstrap CIs for a no-deferral method."""
    rng = np.random.RandomState(seed)
    y_arr = np.array(y_true)
    p = np.array(preds)
    n = len(y_arr)

    boot_acc = np.zeros(B)
    boot_f1 = np.zeros(B)
    boot_prec = np.zeros(B)
    boot_rec = np.zeros(B)
    boot_cvar = np.zeros(B)

    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        y_b = y_arr[idx]
        p_b = p[idx]
        boot_acc[b] = accuracy_score(y_b, p_b)
        boot_f1[b] = f1_score(y_b, p_b, zero_division=0)
        boot_prec[b] = precision_score(y_b, p_b, zero_division=0)
        boot_rec[b] = recall_score(y_b, p_b, zero_division=0)
        losses = (p_b != y_b).astype(float)
        boot_cvar[b] = empirical_cvar(losses, delta)

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "accuracy": (accuracy_score(y_arr, p), ci(boot_acc)),
        "f1": (f1_score(y_arr, p, zero_division=0), ci(boot_f1)),
        "precision": (precision_score(y_arr, p, zero_division=0), ci(boot_prec)),
        "recall": (recall_score(y_arr, p, zero_division=0), ci(boot_rec)),
        "cvar_010": (empirical_cvar((p != y_arr).astype(float), delta), ci(boot_cvar)),
        "deferral_rate": (0.0, (0.0, 0.0)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(val, ci_tuple):
    """Format a metric as 'value [lo, hi]'."""
    return f"{val:.4f} [{ci_tuple[0]:.4f}, {ci_tuple[1]:.4f}]"


def main():
    print("=" * 70)
    print("  New Baselines for Safe-L2D-Fraud Comparison")
    print("=" * 70)

    # ── Phase 1: Data Preparation (same split as existing code) ──
    print("\n[1/7] Loading and preparing data...")
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

    # ── Phase 2: Bayesian + Investigator setup ──
    print("\n[2/7] Setting up Bayesian uncertainty + synthetic investigator...")
    conditionals = compute_class_conditionals(X_train, y_train, BAYESIAN_FEATURES)
    uncertainty = BayesianUncertaintyEstimator(conditionals, prior_fraud)
    investigator = SyntheticInvestigator(base_accuracy=0.85)

    # ── Phase 3: Train Safe-L2D-Fraud (existing system) ──
    print("\n[3/7] Training Safe-L2D-Fraud system...")
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
    system_preds, defer_mask, confidence, entropies, posteriors = \
        system.system_predict(X_test, y_test)
    classifier_preds_safe = np.argmax(system.calibrated_model.predict_proba(X_test), axis=1)

    # ── Phase 4: Train Mozannar-Sontag L2D ──
    print("\n[4/7] Training Mozannar-Sontag L2D baseline...")
    ms_l2d = MozannarSontagL2D(
        preprocessor=build_preprocessor(num_cols, low_cat_cols, med_cat_cols),
        expert_accuracy=0.85,
    )
    ms_l2d.fit(X_train, y_train)
    ms_preds, ms_defer = ms_l2d.system_predict(X_test, y_test, investigator)
    # Get raw classifier predictions for CVaR calc
    ms_raw_preds, _ = ms_l2d.predict(X_test)

    # ── Phase 5: Train Cost-Sensitive XGBoost ──
    print("\n[5/7] Training Cost-Sensitive XGBoost (scale_pos_weight=5)...")
    cs_xgb = CostSensitiveXGBoost(
        preprocessor=build_preprocessor(num_cols, low_cat_cols, med_cat_cols),
        scale_pos_weight=5.0,
    )
    cs_xgb.fit(X_train, y_train)
    cs_preds = cs_xgb.predict(X_test)

    # ── Phase 5b: Vanilla XGBoost (no deferral, no cost-sensitivity) ──
    print("  Training vanilla XGBoost baseline...")
    vanilla_xgb = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, low_cat_cols, med_cat_cols)),
        ("classifier", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )),
    ])
    vanilla_xgb.fit(X_train, y_train)
    vanilla_preds = vanilla_xgb.predict(X_test)

    # ── Phase 5c: Confidence baseline at a fixed threshold ──
    print("  Computing confidence-based deferral baseline...")
    # Use the Safe-L2D calibrated model for confidence baseline
    conf_proba = system.calibrated_model.predict_proba(X_test)
    conf_confidence = np.max(conf_proba, axis=1)
    conf_pred_class = np.argmax(conf_proba, axis=1)
    # Use threshold that gives ~similar deferral rate to Safe-L2D
    conf_thresh = 0.65
    conf_defer = conf_confidence < conf_thresh
    conf_system_preds = conf_pred_class.copy()
    if conf_defer.sum() > 0:
        inv_preds = investigator.predict(
            X_test[conf_defer] if hasattr(X_test, 'loc') else X_test[conf_defer],
            y_test[conf_defer] if hasattr(y_test, 'iloc') else y_test[conf_defer]
        )
        conf_system_preds[conf_defer] = inv_preds

    # ── Phase 6: Bootstrap CIs ──
    print("\n[6/7] Computing bootstrap 95% CIs (B=2000)...")

    print("  Bootstrap: Safe-L2D-Fraud...")
    ci_safe = bootstrap_metrics(y_arr, system_preds, defer_mask, classifier_preds_safe, B=2000)

    print("  Bootstrap: Mozannar-Sontag L2D...")
    ci_ms = bootstrap_metrics(y_arr, ms_preds, ms_defer, ms_raw_preds, B=2000)

    print("  Bootstrap: Confidence baseline...")
    ci_conf = bootstrap_metrics(y_arr, conf_system_preds, conf_defer, conf_pred_class, B=2000)

    print("  Bootstrap: Cost-Sensitive XGBoost...")
    ci_cs = bootstrap_metrics_no_defer(y_arr, cs_preds, B=2000)

    print("  Bootstrap: Vanilla XGBoost...")
    ci_van = bootstrap_metrics_no_defer(y_arr, vanilla_preds, B=2000)

    # ── Phase 7: Print Results ──
    print("\n[7/7] Results\n")

    # Collect all results
    all_results = {
        "Safe-L2D-Fraud (Ours)": ci_safe,
        "Mozannar-Sontag L2D": ci_ms,
        "Confidence Deferral": ci_conf,
        "Cost-Sensitive XGBoost": ci_cs,
        "Vanilla XGBoost": ci_van,
    }

    # ── Table 1: Main comparison ──
    print("=" * 110)
    print("TABLE 1: System-Level Comparison (with 95% Bootstrap CIs, B=2000)")
    print("=" * 110)
    header = f"{'Method':<28} {'Accuracy':<28} {'F1':<28} {'CVaR_0.10':<28} {'Deferral Rate':<28}"
    print(header)
    print("-" * 110)

    for name, ci_dict in all_results.items():
        acc_val, acc_ci = ci_dict["accuracy"]
        f1_val, f1_ci = ci_dict["f1"]
        cvar_val, cvar_ci = ci_dict["cvar_010"]
        def_val, def_ci = ci_dict["deferral_rate"]
        print(f"{name:<28} {fmt(acc_val, acc_ci):<28} {fmt(f1_val, f1_ci):<28} "
              f"{fmt(cvar_val, cvar_ci):<28} {fmt(def_val, def_ci):<28}")

    print("=" * 110)

    # ── Table 2: Detailed metrics for no-deferral baselines ──
    print("\n")
    print("=" * 90)
    print("TABLE 2: No-Deferral Baselines — Detailed Metrics")
    print("=" * 90)
    header2 = f"{'Method':<28} {'Precision':<28} {'Recall':<28} {'F1':<28}"
    print(header2)
    print("-" * 90)

    for name, ci_dict in [("Cost-Sensitive XGBoost", ci_cs), ("Vanilla XGBoost", ci_van)]:
        prec_val, prec_ci = ci_dict["precision"]
        rec_val, rec_ci = ci_dict["recall"]
        f1_val, f1_ci = ci_dict["f1"]
        print(f"{name:<28} {fmt(prec_val, prec_ci):<28} {fmt(rec_val, rec_ci):<28} "
              f"{fmt(f1_val, f1_ci):<28}")

    print("=" * 90)

    # ── Table 3: Deferral methods - what gets deferred ──
    print("\n")
    print("=" * 90)
    print("TABLE 3: Deferral Analysis")
    print("=" * 90)

    defer_methods = {
        "Safe-L2D-Fraud (Ours)": (defer_mask, system_preds, classifier_preds_safe),
        "Mozannar-Sontag L2D": (ms_defer, ms_preds, ms_raw_preds),
        "Confidence Deferral": (conf_defer, conf_system_preds, conf_pred_class),
    }

    header3 = (f"{'Method':<28} {'Deferral Rate':<14} {'Deferred Fraud%':<16} "
               f"{'Sys Acc (non-def)':<18} {'Sys Acc (deferred)':<18} {'Expert Acc (def)':<18}")
    print(header3)
    print("-" * 90)

    for name, (dm_i, sp_i, cp_i) in defer_methods.items():
        def_rate = dm_i.mean()
        # Fraction of deferred that are actually fraud
        if dm_i.sum() > 0:
            fraud_in_deferred = y_arr[dm_i].mean()
            # Expert accuracy on deferred
            expert_acc_def = accuracy_score(y_arr[dm_i], sp_i[dm_i])
        else:
            fraud_in_deferred = 0.0
            expert_acc_def = 0.0

        # Classifier accuracy on non-deferred
        non_defer = ~dm_i
        if non_defer.sum() > 0:
            cls_acc_non_def = accuracy_score(y_arr[non_defer], cp_i[non_defer])
        else:
            cls_acc_non_def = 0.0

        print(f"{name:<28} {def_rate:<14.4f} {fraud_in_deferred:<16.4f} "
              f"{cls_acc_non_def:<18.4f} {expert_acc_def:<18.4f} {expert_acc_def:<18.4f}")

    print("=" * 90)

    # ── Coverage-accuracy curve data points ──
    print("\n")
    print("=" * 70)
    print("TABLE 4: Coverage-Accuracy Operating Points")
    print("=" * 70)

    # Safe-L2D curve
    safe_curve = system.coverage_accuracy_curve(X_test, y_test, method="combined")
    # Confidence curve
    conf_curve = confidence_baseline_curve(
        system.calibrated_model, X_test, y_test, investigator
    )

    # MS-L2D: sweep a threshold on class-2 probability
    ms_proba = ms_l2d.model.predict_proba(X_test)
    ms_coverages = []
    ms_accuracies = []
    ms_f1s = []
    ms_cvars = []

    for thresh in np.linspace(0.0, 0.95, 50):
        # Defer if P(class=2) > thresh
        defer_t = ms_proba[:, 2] > thresh
        preds_t = np.argmax(ms_proba[:, :2], axis=1)  # best of class 0 or 1
        sys_preds_t = preds_t.copy()
        if defer_t.sum() > 0:
            inv_preds_t = investigator.predict(
                X_test[defer_t] if hasattr(X_test, 'loc') else X_test[defer_t],
                y_test[defer_t] if hasattr(y_test, 'iloc') else y_test[defer_t]
            )
            sys_preds_t[defer_t] = inv_preds_t

        cov = (~defer_t).sum() / len(y_arr)
        ms_coverages.append(cov)
        ms_accuracies.append(accuracy_score(y_arr, sys_preds_t))
        ms_f1s.append(f1_score(y_arr, sys_preds_t, zero_division=0))
        non_d_losses = (preds_t[~defer_t] != y_arr[~defer_t]).astype(float) if (~defer_t).sum() > 0 else np.array([0.0])
        ms_cvars.append(empirical_cvar(non_d_losses, 0.10))

    print(f"{'Method':<28} {'Coverage':<12} {'Accuracy':<12} {'F1':<12} {'CVaR_0.10':<12}")
    print("-" * 70)

    # Pick representative coverage points: ~0.70, ~0.80, ~0.90, 1.00
    targets = [0.70, 0.80, 0.90, 1.00]

    for target_cov in targets:
        print(f"\n--- Coverage ~ {target_cov:.2f} ---")

        # Safe-L2D
        valid = ~np.isnan(safe_curve["accuracies"])
        if valid.sum() > 0:
            idx = np.argmin(np.abs(safe_curve["coverages"][valid] - target_cov))
            real_idx = np.where(valid)[0][idx]
            print(f"  {'Safe-L2D-Fraud':<26} {safe_curve['coverages'][real_idx]:<12.4f} "
                  f"{safe_curve['accuracies'][real_idx]:<12.4f} "
                  f"{safe_curve['f1s'][real_idx]:<12.4f} "
                  f"{safe_curve['cvars'][real_idx]:<12.4f}")

        # MS-L2D
        ms_cov_arr = np.array(ms_coverages)
        idx_ms = np.argmin(np.abs(ms_cov_arr - target_cov))
        print(f"  {'Mozannar-Sontag L2D':<26} {ms_coverages[idx_ms]:<12.4f} "
              f"{ms_accuracies[idx_ms]:<12.4f} "
              f"{ms_f1s[idx_ms]:<12.4f} "
              f"{ms_cvars[idx_ms]:<12.4f}")

        # Confidence
        valid_c = ~np.isnan(conf_curve["accuracies"])
        if valid_c.sum() > 0:
            idx_c = np.argmin(np.abs(conf_curve["coverages"][valid_c] - target_cov))
            real_idx_c = np.where(valid_c)[0][idx_c]
            print(f"  {'Confidence Deferral':<26} {conf_curve['coverages'][real_idx_c]:<12.4f} "
                  f"{conf_curve['accuracies'][real_idx_c]:<12.4f} "
                  f"{conf_curve['f1s'][real_idx_c]:<12.4f} "
                  f"{conf_curve['cvars'][real_idx_c]:<12.4f}")

    print("\n" + "=" * 70)

    # ── Summary for paper ──
    print("\n")
    print("=" * 70)
    print("SUMMARY FOR PAPER (point estimates)")
    print("=" * 70)

    print(f"\n{'Method':<28} {'Acc':>8} {'F1':>8} {'CVaR':>8} {'Defer%':>8}")
    print("-" * 60)

    summary_data = [
        ("Safe-L2D-Fraud (Ours)", ci_safe),
        ("Mozannar-Sontag L2D", ci_ms),
        ("Confidence Deferral", ci_conf),
        ("Cost-Sensitive XGBoost", ci_cs),
        ("Vanilla XGBoost", ci_van),
    ]

    for name, ci_dict in summary_data:
        acc = ci_dict["accuracy"][0]
        f1 = ci_dict["f1"][0]
        cvar = ci_dict["cvar_010"][0]
        defer = ci_dict["deferral_rate"][0]
        print(f"{name:<28} {acc:>8.4f} {f1:>8.4f} {cvar:>8.4f} {defer:>8.4f}")

    print("-" * 60)

    # Additional useful info
    print(f"\nDataset: N={len(y)}, N_train={len(y_train)}, N_test={len(y_test)}")
    print(f"Fraud prevalence: {y.mean():.4f}")
    print(f"Expert accuracy (alpha): 0.85")
    print(f"CVaR tail fraction (delta): 0.10")
    print(f"Bootstrap resamples: B=2000")
    print(f"Random state: {RANDOM_STATE}")

    print("\n" + "=" * 70)
    print("  All baselines complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
