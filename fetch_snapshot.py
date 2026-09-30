"""Retrieve the official Toronto Open Data CSVs for a fixed calendar year."""

import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen


EVENTS_URL = (
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
    "996cfe8d-fb35-40ce-b569-698d51fc683b/resource/"
    "0b6e5c52-e993-46d6-8d74-8602ee224457/download/"
    "ttc-subway-delay-data-since-2025.csv"
)
CODES_URL = (
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
    "996cfe8d-fb35-40ce-b569-698d51fc683b/resource/"
    "b2d8f5e0-0997-46b5-8abd-caa685a0290b/download/"
    "code-descriptions.csv"
)


def filter_year_csv(raw_csv, year):
    reader = csv.DictReader(io.StringIO(raw_csv))
    if not reader.fieldnames or "Date" not in reader.fieldnames:
        raise ValueError("source CSV has no Date column")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        try:
            event_date = date.fromisoformat(row["Date"].strip())
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"invalid Date: {row.get('Date')!r}") from exc
        if event_date.year == year:
            writer.writerow(row)
    return output.getvalue()


def download_snapshot(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(EVENTS_URL, timeout=30) as response:
        all_events = response.read().decode("utf-8-sig")
    with urlopen(CODES_URL, timeout=30) as response:
        codes = response.read().decode("utf-8-sig")
    events = filter_year_csv(all_events, 2025)
    paths = {
        "events": output_dir / "ttc_subway_delays_2025.csv",
        "codes": output_dir / "code_descriptions.csv",
    }
    paths["events"].write_text(events, encoding="utf-8")
    paths["codes"].write_text(codes, encoding="utf-8")
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "year": 2025,
        "source_urls": {"events": EVENTS_URL, "codes": CODES_URL},
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }
    (output_dir / "SOURCE_SNAPSHOT.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = download_snapshot(root / "data")
    print(f"Saved official 2025 snapshot to {result['events']}")
