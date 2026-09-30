"""TTC subway incident analysis for the fixed 2025 data snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import fill

import pandas as pd


EVENT_COLUMNS = {"_id", "Date", "Time", "Station", "Code", "Min Delay", "Line"}
CODE_COLUMNS = {"CODE", "DESCRIPTION"}


def _required_columns(frame, required, label):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def _canonical_line(raw):
    value = raw.strip().upper()
    compact = value.replace(" ", "")
    if not compact:
        return "Unknown"
    if "/" in compact or compact.startswith("YU-BD"):
        return "Multiple lines"
    if compact in {"YU", "YUS", "LINE1"}:
        return "Line 1 (YU)"
    if compact == "BD" or compact.startswith("LINE2-"):
        return "Line 2 (BD)"
    if compact == "SHP":
        return "Line 4 (SHP)"
    return "Other / unclassified"


def _clean_description(raw):
    text = raw.strip()
    if "â" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return (
        text.replace("–", " - ").replace("—", " - ")
        .replace("“", '"').replace("”", '"')
        .replace("  ", " ")
    )


def prepare_incidents(events, codes):
    _required_columns(events, EVENT_COLUMNS, "events")
    _required_columns(codes, CODE_COLUMNS, "codes")
    frame = events.copy()
    if frame["_id"].astype(str).duplicated().any():
        raise ValueError("duplicate event IDs in source snapshot")

    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="coerce")
    outside_year = dates.notna() & dates.dt.year.ne(2025)
    if outside_year.any():
        raise ValueError("source snapshot contains records outside 2025")
    times = pd.to_datetime(frame["Time"].astype(str).str.strip(), format="%H:%M", errors="coerce")
    delays = pd.to_numeric(frame["Min Delay"], errors="coerce")
    fractional = delays.notna() & delays.mod(1).ne(0)
    if fractional.any():
        raise ValueError("non-integer delay minutes in source snapshot")
    invalid_date = dates.isna()
    invalid_time = times.isna()
    invalid_delay = delays.isna()
    negative_delay = delays.lt(0)
    excluded = invalid_date | invalid_time | invalid_delay | negative_delay

    clean = frame.loc[~excluded].copy()
    clean["date"] = dates.loc[~excluded]
    clean["hour"] = times.loc[~excluded].dt.hour.astype(int)
    clean["weekday"] = clean["date"].dt.day_name()
    clean["delay_minutes"] = delays.loc[~excluded].astype(int)
    for original, target in (("Line", "line"), ("Station", "station"), ("Code", "code")):
        clean[target] = clean[original].fillna("").astype(str).str.strip().str.upper()
    clean["line"] = clean["line"].map(_canonical_line)
    clean["station"] = clean["station"].replace("", "Unknown")

    code_lookup = codes.copy()
    code_lookup["CODE"] = code_lookup["CODE"].fillna("").astype(str).str.strip().str.upper()
    code_lookup["DESCRIPTION"] = code_lookup["DESCRIPTION"].fillna("").astype(str).map(_clean_description)
    code_lookup = code_lookup.loc[code_lookup["CODE"].ne("")]
    if code_lookup["CODE"].duplicated().any():
        raise ValueError("duplicate delay codes in source lookup")
    descriptions = clean["code"].map(code_lookup.set_index("CODE")["DESCRIPTION"])
    unmapped = descriptions.isna() | descriptions.eq("")
    clean["cause"] = descriptions.where(~unmapped, "Unmapped (" + clean["code"].replace("", "blank") + ")")

    qa = {
        "raw_rows": int(len(frame)),
        "analyzed_rows": int(len(clean)),
        "excluded_rows": int(excluded.sum()),
        "invalid_date_rows": int(invalid_date.sum()),
        "invalid_time_rows": int(invalid_time.sum()),
        "invalid_delay_rows": int(invalid_delay.sum()),
        "negative_delay_rows": int(negative_delay.sum()),
        "zero_delay_rows": int(clean["delay_minutes"].eq(0).sum()),
        "missing_line_rows": int(clean["line"].eq("Unknown").sum()),
        "multiple_line_rows": int(clean["line"].eq("Multiple lines").sum()),
        "other_line_rows": int(clean["line"].eq("Other / unclassified").sum()),
        "missing_station_rows": int(clean["station"].eq("Unknown").sum()),
        "missing_code_rows": int(clean["code"].eq("").sum()),
        "unmapped_code_rows": int(unmapped.sum()),
    }
    return clean, qa


def build_summary(cleaned, qa):
    if cleaned.empty:
        raise ValueError("no analyzable 2025 incidents")

    def grouped(column):
        values = (
            cleaned.groupby(column, dropna=False)
            .agg(incidents=("_id", "size"), delay_minutes=("delay_minutes", "sum"))
            .reset_index()
            .sort_values(["delay_minutes", "incidents", column], ascending=[False, False, True])
        )
        return values.to_dict("records")

    totals = {
        "incidents": int(len(cleaned)),
        "delay_minutes": int(cleaned["delay_minutes"].sum()),
        "positive_delay_incidents": int(cleaned["delay_minutes"].gt(0).sum()),
    }
    summary = {
        "year": 2025,
        "totals": totals,
        "qa": qa,
        "by_cause": grouped("cause"),
        "by_line": grouped("line"),
        "by_hour": grouped("hour"),
        "by_weekday_hour": (
            cleaned.groupby(["weekday", "hour"])
            .agg(incidents=("_id", "size"), delay_minutes=("delay_minutes", "sum"))
            .reset_index()
            .to_dict("records")
        ),
    }
    for group in ("by_cause", "by_line", "by_hour"):
        assert sum(row["incidents"] for row in summary[group]) == totals["incidents"]
        assert sum(row["delay_minutes"] for row in summary[group]) == totals["delay_minutes"]
    return summary


def render_artifacts(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    by_cause = summary["by_cause"]
    top_minutes = sorted(by_cause, key=lambda row: (-row["delay_minutes"], row["cause"]))[:8]
    top_counts = sorted(by_cause, key=lambda row: (-row["incidents"], row["cause"]))[:8]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, rows, metric, color, title in (
        (axes[0], top_minutes, "delay_minutes", "#164b6b", "Recorded delay minutes"),
        (axes[1], top_counts, "incidents", "#bc6c25", "Recorded incidents"),
    ):
        display = list(reversed(rows))
        ax.barh([fill(row["cause"].title(), 34) for row in display], [row[metric] for row in display], color=color)
        ax.set_title(title, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel("Minutes" if metric == "delay_minutes" else "Events")
        ax.grid(axis="x", color="#e6e9ed", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("TTC subway incident causes, 2025", fontsize=16, fontweight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.02, "Source: City of Toronto Open Data. Zero-minute records are included in event counts.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=5)
    fig.savefig(figures / "cause_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    by_line = summary["by_line"]
    order = ["Line 1 (YU)", "Line 2 (BD)", "Line 4 (SHP)", "Multiple lines", "Unknown", "Other / unclassified"]
    lookup = {row["line"]: row for row in by_line}
    line_rows = [lookup[name] for name in order if name in lookup]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, color, title in (
        (axes[0], "incidents", "#bc6c25", "Recorded incidents"),
        (axes[1], "delay_minutes", "#164b6b", "Recorded delay minutes"),
    ):
        ax.barh([row["line"] for row in reversed(line_rows)], [row[metric] for row in reversed(line_rows)], color=color)
        ax.set_title(title, fontweight="bold", loc="left", pad=12)
        ax.grid(axis="x", color="#e6e9ed", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Logged subway incidents by line, 2025", fontsize=16, fontweight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.02, "Counts are not adjusted for ridership or service volume.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9), w_pad=4)
    fig.savefig(figures / "line_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid = [[0] * 24 for _ in weekdays]
    for row in summary["by_weekday_hour"]:
        grid[weekdays.index(row["weekday"])][row["hour"]] = row["delay_minutes"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    plot = ax.imshow(grid, aspect="auto", cmap="YlOrBr")
    ax.set_yticks(range(7), weekdays)
    ax.set_xticks(range(24), [f"{hour:02d}" for hour in range(24)])
    ax.set_xlabel("Hour of recorded incident")
    ax.set_title("Recorded delay minutes by day and hour, 2025", fontsize=16, fontweight="bold", loc="left", pad=15)
    fig.colorbar(plot, ax=ax, label="Recorded delay minutes", shrink=0.88)
    fig.text(0.06, 0.01, "Totals reflect incident logs, not passenger-level delay.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figures / "time_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    totals = summary["totals"]
    qa = summary["qa"]
    cause = by_cause[0]
    line = by_line[0]
    hour = summary["by_hour"][0]
    all_minutes = totals["delay_minutes"]
    cause_share = cause["delay_minutes"] / all_minutes * 100 if all_minutes else 0.0
    line_share = line["delay_minutes"] / all_minutes * 100 if all_minutes else 0.0
    incident_noun = "incident" if totals["incidents"] == 1 else "incidents"
    report = f"""# TTC subway delay incident analysis, 2025

The City of Toronto's log contains **{totals['incidents']:,} recorded {incident_noun}** in 2025. Of these, **{totals['positive_delay_incidents']:,}** have positive delay minutes. The log sums to **{totals['delay_minutes']:,} recorded delay minutes**. These minutes are incident-level entries, not passenger-hours lost.

## What the records show

- **{cause['cause'].title()}** has the largest total: {cause['delay_minutes']:,} minutes across {cause['incidents']:,} incidents ({cause_share:.1f}% of all recorded delay minutes). The cause ranking changes when measured by events instead of minutes; both views are shown below.
- **{line['line']}** has {line['delay_minutes']:,} recorded delay minutes ({line_share:.1f}% of the total) across {line['incidents']:,} logged incidents. This is a workload total, not a reliability rate; lines have different service and ridership levels.
- The highest hourly total is **{hour['hour']:02d}:00-{hour['hour']:02d}:59**, with {hour['delay_minutes']:,} recorded delay minutes across {hour['incidents']:,} incidents. The day-and-hour chart shows where those logged minutes cluster.

![Top causes by two measures](figures/cause_comparison.png)

![Line comparison](figures/line_comparison.png)

![Day and hour pattern](figures/time_heatmap.png)

## Data checks and treatment

| Check | Rows |
| --- | ---: |
| Source snapshot (2025) | {qa['raw_rows']:,} |
| Excluded for invalid date, time, or delay | {qa['excluded_rows']:,} |
| Included in analysis | {qa['analyzed_rows']:,} |
| Included with zero recorded delay minutes | {qa['zero_delay_rows']:,} |
| Cause code without a matching description | {qa['unmapped_code_rows']:,} |
| Missing line, kept as Unknown | {qa['missing_line_rows']:,} |
| Multiple-line labels, kept separate | {qa['multiple_line_rows']:,} |
| Other line labels, kept separate | {qa['other_line_rows']:,} |

Invalid dates, times, non-numeric minutes, and negative minutes are excluded from the analysis. Their diagnostic counts are in `summary.json` and may overlap; the excluded-row total counts each row once. Zero-minute incidents stay in event counts and contribute zero minutes. Missing or unmatched cause codes remain visible rather than being assigned a guessed description. Line aliases such as YUS and LINE 1 are standardized, while mixed-line and non-subway route labels are kept separate. The source cause-description CSV contains damaged punctuation encoding; the analysis repairs those characters for display without changing the code-to-description mapping.

## Interpretation and next question

Start an operational review with causes that have high total recorded minutes, then compare them with incident counts so frequent short events and less frequent long events are not conflated. Before making staffing or service decisions, add train-service volume, ridership, and incident context. This dataset alone cannot estimate a passenger's delay, compare line reliability, or establish that an incident cause led to a particular operational outcome.

Source: [City of Toronto TTC Subway Delay Data](https://open.toronto.ca/dataset/996cfe8d-fb35-40ce-b569-698d51fc683b/), 2025 snapshot. Contains information licensed under the Open Government Licence – Toronto. See the [licence terms](https://open.toronto.ca/open-data-licence/).
"""
    (output_dir / "findings.md").write_text(report, encoding="utf-8")
    return report


def verify_snapshot(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    for name, filename in (("events", "ttc_subway_delays_2025.csv"), ("codes", "code_descriptions.csv")):
        actual = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        if actual != manifest["sha256"][name]:
            raise ValueError(f"snapshot checksum mismatch: {filename}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Analyze the fixed TTC 2025 subway incident snapshot")
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    verify_snapshot(args.data_dir)
    events = pd.read_csv(args.data_dir / "ttc_subway_delays_2025.csv", dtype=str)
    codes = pd.read_csv(args.data_dir / "code_descriptions.csv", dtype=str)
    cleaned, qa = prepare_incidents(events, codes)
    summary = build_summary(cleaned, qa)
    render_artifacts(summary, args.output_dir)
    print(f"Analyzed {summary['totals']['incidents']:,} incidents and wrote {args.output_dir}")


if __name__ == "__main__":
    main()
