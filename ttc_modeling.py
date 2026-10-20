"""Statistical tests and delay-prediction models for the fixed TTC 2025 snapshot.

This module extends ttc_analysis.py without changing it: it reuses the same
cleaning step, then adds a modelling table, a time-based split, and baselines.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ttc_analysis import prepare_incidents, verify_snapshot


RANDOM_SEED = 42
TEST_START = pd.Timestamp("2025-10-01")

# TTC delay codes start with a letter for the responsible area; the second
# letter "U" marks subway. Grouping by the first letter gives broad categories.
CAUSE_CATEGORIES = {
    "E": "Equipment",
    "M": "Miscellaneous",
    "P": "Plant / infrastructure",
    "S": "Security",
    "T": "Transportation",
}

# Features known when an incident is logged. "Min Gap", "Bound" and "Vehicle"
# are excluded on purpose: they are filled in mainly when a train is held, so
# they leak the outcome (see leakage_check). Hour is treated as a category so
# logistic regression can fit a separate effect for each hour.
MODEL_FEATURES = ["line", "weekday", "hour", "cause_code"]
TOP_CODES = 30


def _period(hour):
    if 6 <= hour <= 9:
        return "AM peak"
    if 15 <= hour <= 18:
        return "PM peak"
    if 22 <= hour or hour <= 1:
        return "Late night"
    if 2 <= hour <= 5:
        return "Early morning"
    return "Off-peak day"


def build_model_frame(cleaned, top_codes=TOP_CODES, reference_codes=None):
    """Return one row per incident with features and the binary target.

    reference_codes fixes which cause codes keep their own level; when it is
    None, the most frequent codes in `cleaned` are used. Pass the training-set
    list when transforming test data so no test information is used.
    """
    frame = pd.DataFrame(
        {
            "_id": cleaned["_id"].values,
            "date": cleaned["date"].values,
            "line": cleaned["line"].values,
            "weekday": cleaned["weekday"].values,
            "hour": cleaned["hour"].values,
            "code": cleaned["code"].values,
            "delay_minutes": cleaned["delay_minutes"].values,
        }
    )
    frame["cause_category"] = frame["code"].str[:1].map(CAUSE_CATEGORIES).fillna("Other")
    if reference_codes is None:
        reference_codes = frame["code"].value_counts().head(top_codes).index.tolist()
    # Rare codes are pooled within their category so their broad cause is kept.
    frame["cause_code"] = frame["code"].where(
        frame["code"].isin(reference_codes), "Other " + frame["cause_category"]
    )
    frame["period"] = frame["hour"].map(_period)
    weekday = ~frame["weekday"].isin(["Saturday", "Sunday"])
    frame["is_peak"] = weekday & frame["period"].isin(["AM peak", "PM peak"])
    frame["delayed"] = frame["delay_minutes"].gt(0).astype(int)
    return frame, list(reference_codes)


def time_split(frame, test_start=TEST_START):
    """Train on incidents before test_start, test on incidents from it onward."""
    train = frame.loc[frame["date"] < test_start].copy()
    test = frame.loc[frame["date"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("time split produced an empty train or test set")
    return train, test


def majority_baseline(train, test):
    """Accuracy of always predicting the most common training-set class."""
    majority = int(train["delayed"].mode().iloc[0])
    return {
        "predicted_class": majority,
        "train_accuracy": float(train["delayed"].eq(majority).mean()),
        "test_accuracy": float(test["delayed"].eq(majority).mean()),
    }


def prepare_split(cleaned, test_start=TEST_START, top_codes=TOP_CODES):
    """Time-split the incidents, choosing the kept cause codes from training rows only."""
    train_rows = cleaned.loc[cleaned["date"] < test_start]
    reference_codes = train_rows["code"].value_counts().head(top_codes).index.tolist()
    frame, _ = build_model_frame(cleaned, reference_codes=reference_codes)
    return time_split(frame, test_start)


def _encoder(train, drop_baseline, features=MODEL_FEATURES):
    """One-hot encoder whose levels are ordered by training frequency.

    With drop_baseline, the most common training level of each feature is
    dropped, so each logistic-regression coefficient compares a level with
    that baseline (e.g. a cause code versus SUDP, Disorderly Patron).
    """
    from sklearn.preprocessing import OneHotEncoder

    categories = [train[column].astype(str).value_counts().index.tolist() for column in features]
    return OneHotEncoder(
        categories=categories,
        drop="first" if drop_baseline else None,
        handle_unknown="ignore",
    )


def _features(frame, features=MODEL_FEATURES):
    return frame[list(features)].astype(str)


def build_models(train, seed=RANDOM_SEED, features=MODEL_FEATURES):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return {
        "logistic_regression": make_pipeline(
            _encoder(train, drop_baseline=True, features=features),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        ),
        "random_forest": make_pipeline(
            _encoder(train, drop_baseline=False, features=features),
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
            ),
        ),
    }


def evaluate(y_true, probability, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    )

    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate(train, test, seed=RANDOM_SEED, features=MODEL_FEATURES):
    """Fit both models on training rows; report train and test metrics."""
    results = {}
    fitted = {}
    for name, model in build_models(train, seed, features).items():
        model.fit(_features(train, features), train["delayed"])
        fitted[name] = model
        results[name] = {
            "train": evaluate(train["delayed"], model.predict_proba(_features(train, features))[:, 1]),
            "test": evaluate(test["delayed"], model.predict_proba(_features(test, features))[:, 1]),
        }
    return results, fitted


def logistic_odds_ratios(model):
    """Odds ratio exp(coef) per one-hot level, relative to that feature's baseline."""
    import numpy as np

    encoder, classifier = model.steps[0][1], model.steps[-1][1]
    names = encoder.get_feature_names_out(MODEL_FEATURES)
    baselines = {feature: levels[0] for feature, levels in zip(MODEL_FEATURES, encoder.categories_)}
    table = pd.DataFrame({"feature": names, "coefficient": classifier.coef_[0]})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("coefficient", ascending=False).reset_index(drop=True), baselines


def permutation_importance_by_feature(model, test, seed=RANDOM_SEED, repeats=10):
    """Drop in test ROC AUC when one raw feature's values are shuffled."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, _features(test), test["delayed"], scoring="roc_auc",
        n_repeats=repeats, random_state=seed, n_jobs=1,
    )
    return {
        feature: {"mean_auc_drop": float(mean), "std": float(std)}
        for feature, mean, std in zip(MODEL_FEATURES, result.importances_mean, result.importances_std)
    }


def ablation(train, test, seed=RANDOM_SEED):
    """Test ROC AUC with all features versus without the cause code."""
    context_only = [feature for feature in MODEL_FEATURES if feature != "cause_code"]
    results, _ = train_and_evaluate(train, test, seed, features=context_only)
    return {name: splits["test"]["roc_auc"] for name, splits in results.items()}


SINGLE_LINES = ["Line 1 (YU)", "Line 2 (BD)", "Line 4 (SHP)"]


def chi_square_line(frame, lines=SINGLE_LINES):
    """Chi-square test of independence between line and whether a delay occurred.

    Only single-line labels are tested: "Multiple lines", "Unknown" and other
    labels are logging categories (Multiple lines has no delayed rows at all),
    not lines a passenger rides.
    """
    from scipy.stats import chi2_contingency

    subset = frame.loc[frame["line"].isin(lines)]
    table = pd.crosstab(subset["line"], subset["delayed"]).reindex(index=lines, columns=[0, 1], fill_value=0)
    chi2, p_value, dof, expected = chi2_contingency(table.values)
    n = int(table.values.sum())
    cramers_v = (chi2 / (n * (min(table.shape) - 1))) ** 0.5
    return {
        "n": n,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "min_expected_count": float(expected.min()),
        "delay_rate_by_line": {line: float(table.loc[line, 1] / table.loc[line].sum()) for line in lines},
    }


def mann_whitney_peak(frame):
    """Compare delay minutes in weekday peak vs all other hours, delayed incidents only.

    Zero-minute incidents are left out so the test asks "when a delay happens,
    is it longer in the peak?" rather than mixing in whether a delay happened
    (which the chi-square test and the model already cover).
    """
    from scipy.stats import mannwhitneyu

    delayed = frame.loc[frame["delayed"].eq(1)]
    peak = delayed.loc[delayed["is_peak"], "delay_minutes"]
    off_peak = delayed.loc[~delayed["is_peak"], "delay_minutes"]
    result = mannwhitneyu(peak, off_peak, alternative="two-sided")
    n1, n2 = len(peak), len(off_peak)
    # Probability that a random peak delay is longer than a random off-peak
    # delay (ties count half), and the rank-biserial correlation 2*P - 1.
    prob_peak_longer = result.statistic / (n1 * n2)
    return {
        "n_peak": n1,
        "n_off_peak": n2,
        "median_peak": float(peak.median()),
        "median_off_peak": float(off_peak.median()),
        "mean_peak": float(peak.mean()),
        "mean_off_peak": float(off_peak.mean()),
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "prob_peak_longer": float(prob_peak_longer),
        "rank_biserial": float(2 * prob_peak_longer - 1),
    }


def leakage_check(events):
    """Delay rate split by fields that are filled in after a train is held."""
    delayed = pd.to_numeric(events["Min Delay"], errors="coerce").gt(0)
    blank_bound = events["Bound"].fillna("").str.strip().eq("")
    zero_vehicle = events["Vehicle"].fillna("0").str.strip().eq("0")
    return {
        "delay_rate_blank_bound": float(delayed[blank_bound].mean()),
        "delay_rate_with_bound": float(delayed[~blank_bound].mean()),
        "delay_rate_vehicle_zero": float(delayed[zero_vehicle].mean()),
        "delay_rate_vehicle_known": float(delayed[~zero_vehicle].mean()),
    }


def load_cleaned(data_dir):
    verify_snapshot(data_dir)
    events = pd.read_csv(data_dir / "ttc_subway_delays_2025.csv", dtype=str)
    codes = pd.read_csv(data_dir / "code_descriptions.csv", dtype=str)
    cleaned, _ = prepare_incidents(events, codes)
    return events, cleaned


def run_all(data_dir):
    events, cleaned = load_cleaned(data_dir)
    train, test = prepare_split(cleaned)
    frame = pd.concat([train, test])
    results, fitted = train_and_evaluate(train, test)
    odds, baselines = logistic_odds_ratios(fitted["logistic_regression"])
    return {
        "split": {
            "train_period": "2025-01-01 to 2025-09-30",
            "test_period": "2025-10-01 to 2025-12-31",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_delayed_share": float(train["delayed"].mean()),
            "test_delayed_share": float(test["delayed"].mean()),
        },
        "leakage_check": leakage_check(events),
        "chi_square_line": chi_square_line(frame),
        "mann_whitney_peak": mann_whitney_peak(frame),
        "baseline": majority_baseline(train, test),
        "models": results,
        "test_auc_without_cause_code": ablation(train, test),
        "permutation_importance_lr": permutation_importance_by_feature(fitted["logistic_regression"], test),
        "logistic_baselines": baselines,
        "logistic_odds_ratios": odds.round(4).to_dict("records"),
    }, fitted, test


def render_report(results, fitted, test, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"logistic_regression": "#164b6b", "random_forest": "#bc6c25"}
    for name, model in fitted.items():
        probability = model.predict_proba(_features(test))[:, 1]
        fpr, tpr, _ = roc_curve(test["delayed"], probability)
        auc = results["models"][name]["test"]["roc_auc"]
        axes[0].plot(fpr, tpr, color=colors[name], label=f"{name.replace('_', ' ').title()} (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#999", linestyle="--", label="Chance (AUC 0.500)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("ROC curve, Oct-Dec 2025 test set", fontweight="bold", loc="left")
    axes[0].legend(frameon=False, loc="lower right")

    importance = results["permutation_importance_lr"]
    names = sorted(importance, key=lambda key: importance[key]["mean_auc_drop"])
    axes[1].barh([name.replace("_", " ") for name in names],
                 [importance[name]["mean_auc_drop"] for name in names],
                 xerr=[importance[name]["std"] for name in names], color="#164b6b")
    axes[1].set_xlabel("Drop in test ROC AUC when the feature is shuffled")
    axes[1].set_title("Permutation importance (logistic regression)", fontweight="bold", loc="left")
    axes[1].grid(axis="x", color="#e6e9ed", linewidth=0.7)
    axes[1].set_axisbelow(True)
    fig.tight_layout(w_pad=4)
    fig.savefig(output_dir / "figures" / "model_evaluation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    split, chi, mw = results["split"], results["chi_square_line"], results["mann_whitney_peak"]
    base, lr, rf = results["baseline"], results["models"]["logistic_regression"]["test"], results["models"]["random_forest"]["test"]
    lr_train, rf_train = results["models"]["logistic_regression"]["train"], results["models"]["random_forest"]["train"]
    leak, ablate = results["leakage_check"], results["test_auc_without_cause_code"]
    odds = {row["feature"]: row for row in results["logistic_odds_ratios"]}
    rates = chi["delay_rate_by_line"]

    def row(label, m):
        return (f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | "
                f"{m['f1']:.3f} | {m['roc_auc']:.3f} |")

    def odds_row(code, label):
        record = odds.get(f"cause_code_{code}")
        return f"| {code} | {label} | {record['odds_ratio']:.3f} |" if record else ""

    report = f"""# Statistical tests and delay prediction, TTC subway 2025

This extension asks two questions about the same 25,737 incidents used in the main analysis: (1) do lines and time of day differ in delay outcomes, and (2) how well can an incident's line, day, hour, and cause code predict whether it delayed service (recorded delay minutes > 0)? It does not change the main report's figures. All results are associations in observational data, not causal effects.

## 1. Statistical tests

**Line vs. whether a delay occurred (chi-square test of independence).** Single-line incidents only (n = {chi['n']:,}); "Multiple lines" rows are a logging category with no delayed rows. Delay rates: Line 1 {rates['Line 1 (YU)']:.1%}, Line 2 {rates['Line 2 (BD)']:.1%}, Line 4 {rates['Line 4 (SHP)']:.1%}. Chi-square = {chi['chi2']:.1f}, df = {chi['dof']}, p = {chi['p_value']:.1e}, **Cramér's V = {chi['cramers_v']:.3f}**. The difference is statistically clear but weak: with this sample size a small gap produces a tiny p-value, so the effect size matters more than the p-value.

**Weekday peak vs. other hours, minutes per delayed incident (Mann-Whitney U).** Peak = Monday-Friday 06:00-09:59 and 15:00-18:59. Only incidents with a delay are compared ({mw['n_peak']:,} peak, {mw['n_off_peak']:,} other). Delay minutes are heavily right-skewed, so a rank test is used instead of a t-test. Median {mw['median_peak']:.0f} vs. {mw['median_off_peak']:.0f} minutes; p = {mw['p_value']:.1e}; probability a random peak delay is longer than a random off-peak delay = {mw['prob_peak_longer']:.3f} (rank-biserial r = {mw['rank_biserial']:.3f}). **Peak-hour delays were slightly shorter**, a small-to-moderate effect. The data cannot say why.

## 2. Predicting whether an incident caused a delay

**Time-based split.** Train on {split['train_period']} ({split['train_rows']:,} incidents), test on {split['test_period']} ({split['test_rows']:,}). A random split would let the model learn from incidents that happened after the ones it is tested on; a time split mimics using past data to predict future incidents. The 30 cause codes kept as their own levels are chosen from training rows only; rarer codes are pooled within their broad category (Equipment, Miscellaneous, Plant, Security, Transportation).

**Leakage excluded.** `Bound` and `Vehicle` are almost always blank or zero when no train was held: delayed share is {leak['delay_rate_blank_bound']:.1%} with a blank direction vs. {leak['delay_rate_with_bound']:.1%} with one, and {leak['delay_rate_vehicle_zero']:.1%} with vehicle 0 vs. {leak['delay_rate_vehicle_known']:.1%} otherwise. `Min Gap` is a second delay measure. All three describe the outcome rather than predict it, so none is used.

**Why not accuracy alone.** {1 - split['test_delayed_share']:.1%} of test incidents had no delay, so always predicting "no delay" already scores {base['test_accuracy']:.1%} accuracy with zero recall. Precision (of predicted delays, how many were real), recall (of real delays, how many were caught) and ROC AUC (ranking quality across all thresholds) are reported alongside accuracy. Classification uses a 0.5 probability threshold.

| Model (test set, Oct-Dec) | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always "no delay" baseline | {base['test_accuracy']:.1%} | - | 0.0% | - | 0.500 |
{row('Logistic regression', lr)}
{row('Random forest (300 trees, seed 42)', rf)}

Train-set AUC was {lr_train['roc_auc']:.3f} (logistic) and {rf_train['roc_auc']:.3f} (random forest). The forest fits training data more closely but does not do better on unseen months, so the simpler, interpretable logistic regression is preferred.

![ROC curve and permutation importance](figures/model_evaluation.png)

## 3. What drives the predictions

**The cause code carries almost all of the signal.** Without it (line, weekday, and hour only), test AUC falls to {ablate['logistic_regression']:.3f} (logistic) and {ablate['random_forest']:.3f} (random forest). Permutation importance agrees: shuffling the cause code costs far more AUC than shuffling any other feature.

Logistic-regression odds ratios compare each code with the most common training code, {results['logistic_baselines']['cause_code']} (Disorderly Patron), holding line, weekday and hour fixed:

| Code | Description | Odds ratio |
| --- | --- | ---: |
{odds_row('PUTR', 'Rail related problem')}
{odds_row('MUSAN', 'Unsanitary vehicle')}
{odds_row('PUTWZ', 'Work zone problems - track')}
{odds_row('MUIS', 'Injured/ill customer in station - transported')}
{odds_row('PUMEL', 'Escalator/elevator incident')}
{odds_row('TUNCA', 'No collector available')}

The pattern is mostly **on the track or train vs. in the station**: track, vehicle and on-train medical incidents usually hold trains, while station-only events (escalators, stairways, collector booths) almost never do. This explains the main report's observation that causes rank differently by incident count than by delay minutes: several of the most frequent codes are station events that log an incident but rarely any delay.

## Limitations

- Observational data: odds ratios and test results describe associations, not causes.
- Cause codes may be finalized after an incident, so this model explains the structure of delays more than it would serve as a real-time warning at the moment an incident starts.
- No weather, ridership, service frequency or staffing data, so line and time comparisons are not adjusted for exposure.
- One year of data and one train/test boundary; results for other years may differ.

Reproduce with `.venv/bin/python ttc_modeling.py` (fixed seed 42); numbers are in `results.json`.
"""
    (output_dir / "modeling_report.md").write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Statistical tests and delay models for TTC 2025 incidents")
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "modeling")
    args = parser.parse_args()
    results, fitted, test = run_all(args.data_dir)
    render_report(results, fitted, test, args.output_dir)
    lr = results["models"]["logistic_regression"]["test"]
    print(f"Logistic regression test AUC {lr['roc_auc']:.3f}, accuracy {lr['accuracy']:.3f} "
          f"(baseline {results['baseline']['test_accuracy']:.3f}); wrote {args.output_dir}")


if __name__ == "__main__":
    main()
