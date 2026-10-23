
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "streams_events.csv").exists():
    ROOT = ROOT.parent

EVENTS = ROOT / "streams_events.csv"
STREAMS = ROOT / "streams"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

if not EVENTS.exists():
    raise FileNotFoundError(
        f"Khong tim thay streams_events.csv tai {ROOT}"
    )

df = pd.read_csv(EVENTS)
df["duration"] = df["t_end"] - df["t_start"]

low = df["quality"].eq("low")
review = df["needs_review"].astype(str).str.lower().eq("true")
ignore = df["ignore"].astype(str).str.lower().eq("true")
flagged = low | review

label_stats = {
    "total_events": len(df),
    "quality_low": int(low.sum()),
    "needs_review": int(review.sum()),
    "both_flags": int((low & review).sum()),
    "at_least_one_flag": int(flagged.sum()),
    "ignore_true": int(ignore.sum()),
    "duplicate_event_ids": int(df["event_id"].duplicated().sum()),
    "invalid_intervals": int(
        (df["t_end"] <= df["t_start"]).sum()
    ),
    "missing_cells": int(df.isna().sum().sum()),
}

print("\n=== LABEL QUALITY ===")
for key, value in label_stats.items():
    print(f"{key}: {value}")

print("\nLow quality by dataset:")
print(df.groupby("dataset")["quality"].value_counts())

status_counts = {
    "No flag": int((~low & ~review).sum()),
    "Low only": int((low & ~review).sum()),
    "Review only": int((~low & review).sum()),
    "Both": int((low & review).sum()),
}

plt.figure(figsize=(8, 4.5))
plt.bar(status_counts.keys(), status_counts.values())
plt.title("Fall Event Quality Flags")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.savefig(REPORTS / "04_label_quality.png", dpi=180)
plt.close()

files = sorted(STREAMS.glob("train/*.h5"))
files += sorted(STREAMS.glob("val/*.h5"))

if not files:
    raise FileNotFoundError(
        f"Khong tim thay HDF5 trong {STREAMS}"
    )

results = []
CHUNK = 2048

for path in files:
    split = path.parent.name
    stream_id = f"{split}_{path.stem}"
    row = {
        "stream_id": stream_id,
        "file": str(path.relative_to(ROOT)),
        "error": "",
    }

    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            row["keys"] = ", ".join(keys)

            if "cir" not in f or "t" not in f:
                row["error"] = "Missing cir or t"
                results.append(row)
                continue

            cir = f["cir"]
            t = np.asarray(f["t"][:], dtype=np.float64)

            row["shape"] = str(cir.shape)
            row["dtype"] = str(cir.dtype)

            valid_shape = (
                len(cir.shape) == 3
                and cir.shape[0] == 3
                and cir.shape[2] == 368
            )
            row["valid_shape"] = valid_shape

            if len(cir.shape) != 3:
                row["error"] = "CIR is not 3-dimensional"
                results.append(row)
                continue

            n_frames = cir.shape[1]
            row["frames"] = n_frames
            row["timestamp_length_ok"] = len(t) == n_frames
            row["timestamp_finite"] = bool(np.isfinite(t).all())

            if len(t) >= 2:
                dt = np.diff(t)
                row["duplicate_or_backward_t"] = int(
                    (dt <= 0).sum()
                )
                row["large_time_gaps"] = int(
                    (dt > 0.03).sum()
                )
                row["median_dt"] = float(np.median(dt))
                row["duration_s"] = float(t[-1] - t[0])
                row["t_min"] = float(t[0])
                row["t_max"] = float(t[-1])
            else:
                row["duplicate_or_backward_t"] = 0
                row["large_time_gaps"] = 0
                row["median_dt"] = np.nan
                row["duration_s"] = 0.0
                row["t_min"] = np.nan
                row["t_max"] = np.nan

            row["radar_serials_present"] = "radar_serials" in f
            if "radar_serials" in f:
                row["radar_serials"] = str(
                    f["radar_serials"][:].tolist()
                )

            nan_count = 0
            inf_count = 0
            affected_frames = 0
            zero_frames = 0

            for start in range(0, n_frames, CHUNK):
                end = min(start + CHUNK, n_frames)
                block = cir[:, start:end, :]

                nan_count += int(np.isnan(block).sum())
                inf_count += int(np.isinf(block).sum())

                affected_frames += int(
                    np.any(~np.isfinite(block), axis=(0, 2)).sum()
                )

                zero_frames += int(
                    np.all(block == 0, axis=(0, 2)).sum()
                )

            row["nan_count"] = nan_count
            row["inf_count"] = inf_count
            row["affected_frames"] = affected_frames
            row["all_zero_frames"] = zero_frames
            labels = df[df["stream_id"] == stream_id]
            row["event_count"] = len(labels)

            if len(t):
                outside = (
                    (labels["t_start"] < t[0] - 0.02)
                    | (labels["t_end"] > t[-1] + 0.02)
                )
                row["events_outside_stream"] = int(outside.sum())
            else:
                row["events_outside_stream"] = np.nan

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    results.append(row)
    print("Checked:", stream_id)

quality = pd.DataFrame(results)
quality.to_csv(
    REPORTS / "04_data_quality_per_stream.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(
    [label_stats]
).to_csv(
    REPORTS / "04_label_quality_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== HDF5 DATA QUALITY ===")
columns = [
    "stream_id", "valid_shape", "frames",
    "nan_count", "inf_count", "affected_frames",
    "duplicate_or_backward_t", "large_time_gaps",
    "events_outside_stream", "error"
]
print(quality.reindex(columns=columns).to_string(index=False))

metric_cols = [
    "nan_count", "inf_count", "affected_frames",
    "duplicate_or_backward_t", "large_time_gaps",
    "events_outside_stream"
]
totals = quality.reindex(columns=metric_cols).apply(
    pd.to_numeric, errors="coerce"
).sum()

text = f"""# DATA QUALITY ANALYSIS

## 1. Label quality
- Total fall events: {len(df)}
- Quality low: {low.sum()}
- Needs review: {review.sum()}
- Both flags: {(low & review).sum()}
- At least one flag: {flagged.sum()}
- Explicit ignore: {ignore.sum()}
- Duplicate event IDs: {label_stats['duplicate_event_ids']}
- Invalid event intervals: {label_stats['invalid_intervals']}
- Missing CSV cells: {label_stats['missing_cells']}

## 2. HDF5 quality
- Files scanned: {len(quality)}
- NaN values: {int(totals['nan_count'])}
- Inf values: {int(totals['inf_count'])}
- Affected frames: {int(totals['affected_frames'])}
- Duplicate/backward timestamp steps: {int(totals['duplicate_or_backward_t'])}
- Time gaps above 0.03s: {int(totals['large_time_gaps'])}
- Events outside stream bounds: {int(totals['events_outside_stream'])}

## 3. Cautions
- All-zero frames may be legitimate, not necessarily errors.
- Regular 50 Hz timestamps do not prove raw measurements are complete.
- Quality=low is an annotation-quality flag, not a NaN indicator.
- Verify the per-stream CSV and any scan errors before concluding.
"""

(REPORTS / "04_data_quality_report.md").write_text(
    text, encoding="utf-8"
)

print("\nSaved reports to:", REPORTS)
