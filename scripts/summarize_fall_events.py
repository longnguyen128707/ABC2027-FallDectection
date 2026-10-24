"""Summarize ground-truth fall events per stream and plot duration histograms."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FALL_TYPES = ("Chute01", "Chute02", "Chute03", "Chute04")
CSV_FIELDS = (
    "stream_id",
    "split",
    "event_count",
    *(f"{fall_type.lower()}_count" for fall_type in FALL_TYPES),
    "duration_mean_s",
    "duration_median_s",
    "duration_std_s",
)


def load_events(path: Path) -> dict[str, list[dict[str, str]]]:
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    required = {"stream_id", "t_start", "t_end", "fall_type"}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            events[row["stream_id"]].append(row)
    return events


def stream_ids() -> list[str]:
    paths = sorted((ROOT / "streams").glob("*/*.h5"))
    ids = [f"{path.parent.name}_{path.stem.rsplit('_', 1)[-1]}" for path in paths]
    if not ids:
        raise FileNotFoundError(f"No stream HDF5 files found under {ROOT / 'streams'}")
    return ids


def summarize_stream(stream_id: str, events: list[dict[str, str]]) -> dict[str, object]:
    durations = []
    counts = {fall_type: 0 for fall_type in FALL_TYPES}
    for event in events:
        start = float(event["t_start"])
        end = float(event["t_end"])
        duration = end - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{stream_id}: invalid event interval {start} to {end}")
        durations.append(duration)
        fall_type = event["fall_type"]
        if fall_type not in counts:
            raise ValueError(f"{stream_id}: unknown fall_type {fall_type!r}")
        counts[fall_type] += 1

    row: dict[str, object] = {
        "stream_id": stream_id,
        "split": stream_id.split("_", 1)[0],
        "event_count": len(events),
    }
    row.update({f"{fall_type.lower()}_count": counts[fall_type] for fall_type in FALL_TYPES})
    if durations:
        row.update(
            {
                "duration_mean_s": round(float(np.mean(durations)), 3),
                "duration_median_s": round(float(np.median(durations)), 3),
                "duration_std_s": round(float(np.std(durations)), 3),
            }
        )
    else:
        row.update({"duration_mean_s": "", "duration_median_s": "", "duration_std_s": ""})
    return row


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_duration_histogram(events_by_stream: dict[str, list[dict[str, str]]], path: Path) -> None:
    durations_by_split: dict[str, list[float]] = defaultdict(list)
    for stream_id, events in events_by_stream.items():
        split = stream_id.split("_", 1)[0]
        for event in events:
            durations_by_split[split].append(float(event["t_end"]) - float(event["t_start"]))

    if not durations_by_split:
        raise ValueError("No ground-truth events found; cannot plot duration histogram")

    all_durations = [duration for values in durations_by_split.values() for duration in values]
    bins = np.histogram_bin_edges(all_durations, bins="fd")
    if len(bins) < 2:
        bins = np.linspace(min(all_durations) - 0.5, max(all_durations) + 0.5, 11)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for split in ("train", "val"):
        durations = durations_by_split.get(split, [])
        if durations:
            axis.hist(durations, bins=bins, alpha=0.55, label=f"{split} (n={len(durations)})")
    axis.set_xlabel("Fall event duration (s)")
    axis.set_ylabel("Number of events")
    axis.set_title("Ground-truth fall duration distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "outputs" / "fall_event_summary_by_stream.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--histogram",
        type=Path,
        default=ROOT / "outputs" / "fall_duration_histogram.png",
        help="Output histogram PNG path.",
    )
    args = parser.parse_args()

    events_by_stream = load_events(ROOT / "streams_events.csv")
    rows = [
        summarize_stream(stream_id, events_by_stream.get(stream_id, []))
        for stream_id in stream_ids()
    ]
    write_summary(args.csv, rows)
    plot_duration_histogram(events_by_stream, args.histogram)
    print(f"Wrote {len(rows)} stream summaries to {args.csv}")
    print(f"Wrote duration histogram to {args.histogram}")


if __name__ == "__main__":
    main()
