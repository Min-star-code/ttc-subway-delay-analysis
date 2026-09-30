import unittest
import json
from pathlib import Path
import tempfile

import pandas as pd

from ttc_analysis import build_summary, prepare_incidents, render_artifacts


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame(
            [
                {"_id": "1", "Date": "2025-01-01", "Time": "02:10", "Day": "Wednesday", "Station": " BATHURST ", "Code": "A", "Min Delay": "5", "Min Gap": "9", "Bound": "E", "Line": "BD", "Vehicle": "1"},
                {"_id": "2", "Date": "2025-01-01", "Time": "02:30", "Day": "Wednesday", "Station": "", "Code": "B", "Min Delay": "0", "Min Gap": "0", "Bound": "", "Line": "", "Vehicle": "2"},
                {"_id": "3", "Date": "2025-01-02", "Time": "25:00", "Day": "Thursday", "Station": "DUFFERIN", "Code": "A", "Min Delay": "3", "Min Gap": "4", "Bound": "", "Line": "BD", "Vehicle": "3"},
                {"_id": "4", "Date": "2025-01-03", "Time": "13:00", "Day": "Friday", "Station": "DUFFERIN", "Code": "A", "Min Delay": "-1", "Min Gap": "0", "Bound": "", "Line": "BD", "Vehicle": "4"},
                {"_id": "5", "Date": "2025-01-04", "Time": "14:15", "Day": "Saturday", "Station": "DUFFERIN", "Code": "X", "Min Delay": "8", "Min Gap": "0", "Bound": "", "Line": "YU", "Vehicle": "5"},
            ]
        )
        self.codes = pd.DataFrame(
            [{"CODE": "A", "DESCRIPTION": "EQUIPMENT"}, {"CODE": "B", "DESCRIPTION": "OPERATIONS"}]
        )

    def test_keeps_zero_minutes_and_unmapped_code_but_excludes_invalid_rows(self):
        cleaned, qa = prepare_incidents(self.events, self.codes)
        self.assertEqual(cleaned["_id"].tolist(), ["1", "2", "5"])
        self.assertEqual(cleaned["delay_minutes"].tolist(), [5, 0, 8])
        self.assertEqual(cleaned["cause"].tolist(), ["EQUIPMENT", "OPERATIONS", "Unmapped (X)"])
        self.assertEqual(cleaned["line"].tolist(), ["Line 2 (BD)", "Unknown", "Line 1 (YU)"])
        self.assertEqual(cleaned["station"].tolist(), ["BATHURST", "Unknown", "DUFFERIN"])
        self.assertEqual(qa["raw_rows"], 5)
        self.assertEqual(qa["analyzed_rows"], 3)
        self.assertEqual(qa["invalid_time_rows"], 1)
        self.assertEqual(qa["negative_delay_rows"], 1)
        self.assertEqual(qa["zero_delay_rows"], 1)
        self.assertEqual(qa["unmapped_code_rows"], 1)
        self.assertEqual(qa["missing_line_rows"], 1)

    def test_rejects_duplicate_source_ids(self):
        events = pd.concat([self.events.iloc[[0]], self.events.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate event IDs"):
            prepare_incidents(events, self.codes)

    def test_rejects_records_outside_frozen_year(self):
        events = self.events.iloc[[0]].copy()
        events.loc[0, "Date"] = "2026-01-01"
        with self.assertRaisesRegex(ValueError, "outside 2025"):
            prepare_incidents(events, self.codes)

    def test_counts_and_minutes_reconcile_independently(self):
        cleaned, qa = prepare_incidents(self.events, self.codes)
        summary = build_summary(cleaned, qa)
        self.assertEqual(summary["totals"], {"incidents": 3, "delay_minutes": 13, "positive_delay_incidents": 2})
        self.assertEqual(sum(row["incidents"] for row in summary["by_cause"]), 3)
        self.assertEqual(sum(row["delay_minutes"] for row in summary["by_cause"]), 13)
        self.assertEqual(sum(row["incidents"] for row in summary["by_line"]), 3)
        self.assertEqual(sum(row["delay_minutes"] for row in summary["by_line"]), 13)
        self.assertEqual(sum(row["incidents"] for row in summary["by_hour"]), 3)
        self.assertEqual(sum(row["delay_minutes"] for row in summary["by_hour"]), 13)

    def test_repairs_source_text_and_groups_line_aliases_without_reassigning_routes(self):
        events = self.events.iloc[[0, 1, 4]].copy().reset_index(drop=True)
        events.loc[0, "Line"] = "YUS"
        events.loc[1, "Line"] = "YU / BD"
        events.loc[2, "Line"] = "95 YORK MILLS"
        codes = self.codes.copy()
        codes.loc[0, "DESCRIPTION"] = "SIGNAL â\x80\x93 NO TROUBLE FOUND"
        cleaned, qa = prepare_incidents(events, codes)
        self.assertEqual(cleaned["line"].tolist(), ["Line 1 (YU)", "Multiple lines", "Other / unclassified"])
        self.assertEqual(cleaned["cause"].iloc[0], "SIGNAL - NO TROUBLE FOUND")
        self.assertEqual(qa["multiple_line_rows"], 1)
        self.assertEqual(qa["other_line_rows"], 1)

    def test_rejects_fractional_delay_instead_of_truncating_minutes(self):
        events = self.events.iloc[[0]].copy()
        events.loc[0, "Min Delay"] = "2.5"
        with self.assertRaisesRegex(ValueError, "non-integer delay"):
            prepare_incidents(events, self.codes)

    def test_report_and_charts_use_reconciled_totals(self):
        cleaned, qa = prepare_incidents(self.events, self.codes)
        summary = build_summary(cleaned, qa)
        with tempfile.TemporaryDirectory() as folder:
            render_artifacts(summary, Path(folder))
            report = (Path(folder) / "findings.md").read_text(encoding="utf-8")
            saved = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("3 recorded incidents", report)
            self.assertIn("13 recorded delay minutes", report)
            self.assertEqual(saved["totals"]["delay_minutes"], 13)
            for name in ("cause_comparison.png", "line_comparison.png", "time_heatmap.png"):
                self.assertGreater((Path(folder) / "figures" / name).stat().st_size, 1000)

    def test_report_handles_year_with_only_zero_minute_records(self):
        cleaned, qa = prepare_incidents(self.events.iloc[[1]], self.codes)
        summary = build_summary(cleaned, qa)
        with tempfile.TemporaryDirectory() as folder:
            report = render_artifacts(summary, Path(folder))
            self.assertIn("1 recorded incident", report)
            self.assertIn("0 recorded delay minutes", report)


if __name__ == "__main__":
    unittest.main()
