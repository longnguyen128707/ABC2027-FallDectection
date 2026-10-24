"""Copy of the starter notebook extended to analyze every public stream.

This script keeps the same dataset assumptions as the notebook but adds a
stream-by-stream analysis pass for all public train/validation streams, then
writes a baseline validation submission and evaluates it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\ASUS\Downloads\abc2027_uwb_train_v1\abc2027_uwb_train_v1")
ROOT = REPO

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abc2027.baseline import energy_events  # noqa: E402
from abc2027.evaluate import evaluate, load_ground_truth, stream_durations, validate_predictions  # noqa: E402

EVENT_STYLES = {
    "normal": ("tab:green", "normal"),
    "low": ("tab:orange", "low quality"),
    "needs_review": ("tab:red", "needs review"),
    "low_needs_review": ("tab:purple", "low quality + needs review"),
}


def stream_id_from_path(path: Path) -> str:
    """Map a stream_XX.h5 file to its public stream id, e.g. val_01."""
    stem = path.stem
    split = path.parent.name
    index = int(stem.split("_")[-1])
    return f"{split}_{index:02d}"


def event_rows_for_stream(rows: list[dict], stream_id: str) -> list[dict]:
    return [row for row in rows if row.get("stream_id") == stream_id]


def event_category(row: dict) -> str:
    low_quality = row.get("quality", "").strip().lower() == "low"
    needs_review = row.get("needs_review", "").strip().lower() in {"1", "true", "yes"}
    if low_quality and needs_review:
        return "low_needs_review"
    if low_quality:
        return "low"
    if needs_review:
        return "needs_review"
    return "normal"


def plot_stream_energy(stream_path: Path, event_rows: list[dict], output_dir: Path) -> Path:
    with h5py.File(stream_path, "r") as handle:
        cir = handle["cir"][:]
        t = handle["t"][:]

    energy = np.mean(np.sum(cir.astype(np.float64) ** 2, axis=2), axis=0)
    stream_id = stream_id_from_path(stream_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stream_id}_energy.png"

    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(t[::10], energy[::10], color="black", linewidth=0.55, label="CIR energy")
    used_categories = set()
    for row in event_rows:
        category = event_category(row)
        color, label = EVENT_STYLES[category]
        axis.axvspan(
            float(row["t_start"]),
            float(row["t_end"]),
            color=color,
            alpha=0.3,
            label=label if category not in used_categories else None,
        )
        used_categories.add(category)

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean radar energy (mean sum of CIR²)")
    axis.set_title(f"{stream_id}: energy with annotated fall events (n={len(event_rows)})")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


def summarize_stream(stream_path: Path, event_rows: list[dict]) -> dict:
    with h5py.File(stream_path, "r") as handle:
        cir = handle["cir"][:]
        t = handle["t"][:]

    energy = np.mean(np.sum(cir.astype(np.float64) ** 2, axis=2), axis=0)
    categories = {category: 0 for category in EVENT_STYLES}
    for row in event_rows:
        categories[event_category(row)] += 1

    stats = {
        "stream_id": stream_id_from_path(stream_path),
        "path": str(stream_path),
        "duration_s": float(cir.shape[1] / 50.0),
        "n_frames": int(cir.shape[1]),
        "n_events": len(event_rows),
        "event_categories": categories,
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "energy_min": float(np.min(energy)),
        "energy_p90": float(np.percentile(energy, 90)),
        "energy_p95": float(np.percentile(energy, 95)),
        "energy_p99": float(np.percentile(energy, 99)),
        "max_energy": float(np.max(energy)),
        "events": [
            {
                "event_id": row["event_id"],
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "quality": row.get("quality", ""),
                "needs_review": row.get("needs_review", "").strip().lower() in {"1", "true", "yes"},
                "ignore": row.get("ignore", "").strip().lower() in {"1", "true", "yes"},
                "category": event_category(row),
            }
            for row in event_rows
        ],
    }
    return stats


def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stream_id": row["stream_id"],
                    "t_start": f"{row['t_start']:.3f}",
                    "t_end": f"{row['t_end']:.3f}",
                    "score": f"{row['score']:.6f}",
                    "t_alarm": f"{row['t_alarm']:.3f}",
                }
            )


def analyze_all_streams() -> dict:
    events_path = ROOT / "streams_events.csv"
    with events_path.open(encoding="utf-8", newline="") as handle:
        all_events = list(csv.DictReader(handle))

    stats = []
    for split in ("train", "val"):
        folder = ROOT / "streams" / split
        for stream_path in sorted(folder.glob("stream_*.h5")):
            stream_id = stream_id_from_path(stream_path)
            gt_rows = event_rows_for_stream(all_events, stream_id)
            stream_stats = summarize_stream(stream_path, gt_rows)
            plot_path = plot_stream_energy(
                stream_path,
                gt_rows,
                ROOT / "outputs" / "stream_energy_plots",
            )
            stream_stats["energy_plot"] = str(plot_path)
            stats.append(stream_stats)

    return {"streams": stats, "n_streams": len(stats), "n_events": sum(item["n_events"] for item in stats)}


def build_baseline_submission(split: str = "val") -> list[dict]:
    folder = ROOT / "streams" / split
    rows: list[dict] = []
    for stream_path in sorted(folder.glob("stream_*.h5")):
        stream_id = stream_id_from_path(stream_path)
        with h5py.File(stream_path, "r") as handle:
            cir = handle["cir"][:]
        detected = energy_events(cir, percentile=98.0, min_duration=0.6, merge_gap=0.8)
        for event in detected:
            rows.append(
                {
                    "stream_id": stream_id,
                    "t_start": float(event["t_start"]),
                    "t_end": float(event["t_end"]),
                    "score": float(event["score"]),
                    "t_alarm": float(event["t_alarm"]),
                }
            )
    return rows


def main() -> None:
    print("Analyzing public streams...")
    analysis = analyze_all_streams()
    for item in analysis["streams"]:
        print(
            f"{item['stream_id']:>7} | duration={item['duration_s']:7.1f}s | "
            f"events={item['n_events']:>2} | normal={item['event_categories']['normal']:>2} "
            f"low={item['event_categories']['low']:>2} "
            f"review={item['event_categories']['needs_review'] + item['event_categories']['low_needs_review']:>2} | "
            f"energy_mean={item['energy_mean']:.4e} | energy_p99={item['energy_p99']:.4e}"
        )

    print(f"\nTotal public streams analyzed: {analysis['n_streams']}")
    print(f"Total annotated falls: {analysis['n_events']}")
    print(f"Energy plots saved under: {ROOT / 'outputs' / 'stream_energy_plots'}")

    val_rows = build_baseline_submission(split="val")
    out_path = ROOT / "outputs" / "all_val_baseline_submission_quality_analysis.csv"
    write_predictions_csv(out_path, val_rows)
    print(f"Saved baseline predictions to: {out_path}")

    durations = stream_durations(ROOT / "streams")
    gt = load_ground_truth(ROOT / "streams_events.csv", ignore_flagged=True)
    predictions = validate_predictions(out_path, durations)
    report = evaluate(gt, predictions, durations, theta=0.3)

    print("\nValidation report (ignore_flagged=True):")
    print(json.dumps({
        "ranking_score": report["ranking_score"],
        "primary_theta_0_3": report["primary"],
    }, indent=2))

    summary_path = ROOT / "outputs" / "stream_quality_analysis_summary.json"
    summary_path.write_text(json.dumps({"analysis": analysis, "report": report}, indent=2), encoding="utf-8")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
