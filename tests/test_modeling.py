import unittest

import pandas as pd

from ttc_modeling import (
    build_model_frame,
    chi_square_line,
    evaluate,
    logistic_odds_ratios,
    majority_baseline,
    mann_whitney_peak,
    prepare_split,
    time_split,
    train_and_evaluate,
)


def _cleaned(rows):
    frame = pd.DataFrame(rows, columns=["_id", "date", "line", "hour", "code", "delay_minutes"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["weekday"] = frame["date"].dt.day_name()
    return frame


class ModelFrameTests(unittest.TestCase):
    def setUp(self):
        self.cleaned = _cleaned(
            [
                ("1", "2025-03-03", "Line 1 (YU)", 8, "MUSAN", 5),   # Monday AM peak
                ("2", "2025-03-08", "Line 2 (BD)", 8, "MUSAN", 0),   # Saturday 08:00
                ("3", "2025-09-30", "Line 1 (YU)", 13, "EUSC", 0),
                ("4", "2025-10-01", "Line 2 (BD)", 17, "PUMST", 3),  # first test day
                ("5", "2025-12-31", "Line 4 (SHP)", 23, "ZZZZ", 0),
            ]
        )

    def test_target_categories_and_weekday_only_peak(self):
        frame, codes = build_model_frame(self.cleaned, top_codes=2)
        self.assertEqual(frame["delayed"].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(frame["cause_category"].tolist(),
                         ["Miscellaneous", "Miscellaneous", "Equipment", "Plant / infrastructure", "Other"])
        self.assertEqual(frame["is_peak"].tolist(), [True, False, False, True, False])
        self.assertEqual(codes[0], "MUSAN")
        self.assertEqual(frame.loc[4, "cause_code"], "Other Other")

    def test_reference_codes_from_training_are_reused(self):
        frame, _ = build_model_frame(self.cleaned, reference_codes=["EUSC"])
        self.assertEqual(frame["cause_code"].tolist(),
                         ["Other Miscellaneous", "Other Miscellaneous", "EUSC",
                          "Other Plant / infrastructure", "Other Other"])

    def test_prepare_split_picks_codes_from_training_rows_only(self):
        _, test = prepare_split(self.cleaned, top_codes=1)
        # PUMST appears only in the test period, so it must be pooled.
        self.assertEqual(test["cause_code"].tolist(), ["Other Plant / infrastructure", "Other Other"])

    def test_time_split_has_no_overlap_and_baseline_uses_training_majority(self):
        frame, _ = build_model_frame(self.cleaned)
        train, test = time_split(frame)
        self.assertEqual(train["_id"].tolist(), ["1", "2", "3"])
        self.assertEqual(test["_id"].tolist(), ["4", "5"])
        self.assertLess(train["date"].max(), test["date"].min())
        baseline = majority_baseline(train, test)
        self.assertEqual(baseline["predicted_class"], 0)
        self.assertAlmostEqual(baseline["test_accuracy"], 0.5)


class StatisticalTestTests(unittest.TestCase):
    def _frame(self, line_counts):
        rows = []
        for line, (not_delayed, delayed) in line_counts.items():
            rows += [{"line": line, "delayed": 0}] * not_delayed
            rows += [{"line": line, "delayed": 1}] * delayed
        return pd.DataFrame(rows)

    def test_chi_square_detects_no_association_for_equal_rates(self):
        frame = self._frame({"Line 1 (YU)": (60, 40), "Line 2 (BD)": (60, 40), "Line 4 (SHP)": (30, 20)})
        result = chi_square_line(frame)
        self.assertAlmostEqual(result["chi2"], 0.0)
        self.assertAlmostEqual(result["p_value"], 1.0)
        self.assertAlmostEqual(result["cramers_v"], 0.0)

    def test_chi_square_ignores_non_line_labels_and_reports_perfect_association(self):
        frame = self._frame({"Line 1 (YU)": (0, 50), "Line 2 (BD)": (50, 0), "Line 4 (SHP)": (0, 50),
                             "Multiple lines": (500, 0)})
        result = chi_square_line(frame)
        self.assertEqual(result["n"], 150)
        self.assertAlmostEqual(result["cramers_v"], 1.0)
        self.assertLess(result["p_value"], 1e-10)

    def test_mann_whitney_uses_delayed_rows_and_reports_effect_size(self):
        frame = pd.DataFrame(
            {
                "delayed": [1, 1, 1, 1, 1, 1, 0],
                "is_peak": [True, True, True, False, False, False, True],
                "delay_minutes": [10, 11, 12, 1, 2, 3, 0],
            }
        )
        result = mann_whitney_peak(frame)
        self.assertEqual((result["n_peak"], result["n_off_peak"]), (3, 3))
        self.assertAlmostEqual(result["prob_peak_longer"], 1.0)
        self.assertAlmostEqual(result["rank_biserial"], 1.0)
        self.assertEqual(result["median_peak"], 11.0)


class ModelTests(unittest.TestCase):
    def _synthetic(self):
        # Code A is always delayed and code B never is; line and time add noise.
        rows = []
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        for i, date in enumerate(dates):
            for j, code in enumerate(["EUA", "MUB"]):
                line = ["Line 1 (YU)", "Line 2 (BD)"][(i + j) % 2]
                rows.append((f"{i}-{j}", date, line, (i * 5 + j) % 24, code, 4 if code == "EUA" else 0))
        return _cleaned(rows)

    def test_metrics_match_hand_counts(self):
        metrics = evaluate(pd.Series([1, 1, 0, 0]), pd.Series([0.9, 0.4, 0.6, 0.1]))
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)

    def test_models_learn_signal_and_are_reproducible(self):
        train, test = prepare_split(self._synthetic())
        first, fitted = train_and_evaluate(train, test)
        second, _ = train_and_evaluate(train, test)
        self.assertEqual(first, second)
        for name in ("logistic_regression", "random_forest"):
            self.assertGreater(first[name]["test"]["roc_auc"], 0.99)
        odds, baselines = logistic_odds_ratios(fitted["logistic_regression"])
        self.assertIn(baselines["cause_code"], {"EUA", "MUB"})
        self.assertEqual(len(odds), 1 + 1 + 6 + 23)  # one-hot levels minus each baseline


if __name__ == "__main__":
    unittest.main()
