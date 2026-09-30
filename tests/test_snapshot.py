import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fetch_snapshot import filter_year_csv
from ttc_analysis import verify_snapshot


class SnapshotTests(unittest.TestCase):
    def test_selects_full_calendar_year_and_preserves_columns(self):
        raw = "_id,Date,Time,Min Delay\n1,2025-01-01,00:00,0\n2,2025-12-31,23:59,7\n3,2026-01-01,00:00,8\n"
        result = list(csv.DictReader(io.StringIO(filter_year_csv(raw, 2025))))
        self.assertEqual([row["_id"] for row in result], ["1", "2"])
        self.assertEqual(result[1]["Min Delay"], "7")

    def test_rejects_malformed_source_date(self):
        with self.assertRaisesRegex(ValueError, "invalid Date"):
            filter_year_csv("_id,Date\n1,2025-13-01\n", 2025)

    def test_detects_changed_snapshot_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            events = root / "ttc_subway_delays_2025.csv"
            codes = root / "code_descriptions.csv"
            events.write_text("Date\n2025-01-01\n", encoding="utf-8")
            codes.write_text("CODE,DESCRIPTION\nA,Signal\n", encoding="utf-8")
            manifest = {"sha256": {"events": hashlib.sha256(events.read_bytes()).hexdigest(), "codes": hashlib.sha256(codes.read_bytes()).hexdigest()}}
            (root / "SOURCE_SNAPSHOT.json").write_text(json.dumps(manifest), encoding="utf-8")
            verify_snapshot(root)
            events.write_text("Date\n2026-01-01\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_snapshot(root)


if __name__ == "__main__":
    unittest.main()
