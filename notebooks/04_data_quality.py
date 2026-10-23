
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

REPORTS.mkdir(parents=True, exist_ok=True)

if not EVENTS.exists():
    raise FileNotFoundError(
        f"Cannot find streams_events.csv in {ROOT}"
    )

df = pd.read_csv(EVENTS)
df["duration"] = df["t_end"] - df["t_start"]

low = df["quality"].eq("low")

review = (
    df["needs_review"]
    .astype(str)
    .str.lower()
    .eq("true")
)

ignore = (
    df["ignore"]
    .astype(str)
    .str.lower()
    .eq("true")
)

flagged = low | review

label_stats = {
    "total_events": len(df),
    "quality_low": int(low.sum()),
    "needs_review": int(review.sum()),
    "both_flags": int((low & review).sum()),
    "at_least_one_flag": int(flagged.sum()),
    "ignore_true": int(ignore.sum()),
    "duplicate_event_ids": int(
        df["event_id"].duplicated().sum()
    ),
    "invalid_intervals": int(
        (df["t_end"] <= df["t_start"]).sum()
    ),
    "missing_cells": int(
        df.isna().sum().sum()
    ),
}

print("\n========== LABEL QUALITY ==========")

for key, value in label_stats.items():
    print(f"{key}: {value}")

print("\nQuality distribution by dataset:")

print(
    df.groupby("dataset")["quality"].value_counts()
)

status_counts = {
    "No flag": int((~low & ~review).sum()),
    "Low only": int((low & ~review).sum()),
    "Review only": int((~low & review).sum()),
    "Both": int((low & review).sum()),
}

plt.figure(figsize=(8, 4.5))

bars = plt.bar(
    status_counts.keys(),
    status_counts.values()
)

plt.title("Fall Event Quality Flags")
plt.xlabel("Quality Category")
plt.ylabel("Number of Events")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(status_counts.values()) * 1.15)
plt.tight_layout()

plt.savefig(
    REPORTS / "04_label_quality.png",
    dpi=180
)

plt.close()


files = sorted(STREAMS.glob("train/*.h5"))
files += sorted(STREAMS.glob("val/*.h5"))

if not files:
    raise FileNotFoundError(
        f"No HDF5 files found in {STREAMS}"
    )

print("\n========== HDF5 FILES ==========")
print("Number of files:", len(files))

results = []
CHUNK = 2048



for path in files:

    split = path.parent.name
    stream_number = path.stem.removeprefix("stream_")
    stream_id = f"{split}_{stream_number}"

    labels = df[df["stream_id"] == stream_id]

    row = {
        "stream_id": stream_id,
        "file": str(path.relative_to(ROOT)),
        "error": "",
        "event_count": len(labels),
    }

    print(f"\nChecking {stream_id} ...", flush=True)

    try:
        with h5py.File(path, "r") as f:

            row["keys"] = ", ".join(f.keys())

            if "cir" not in f or "t" not in f:
                row["error"] = "Missing cir or t"
                results.append(row)
                continue

            cir = f["cir"]
            t = np.asarray(
                f["t"][:],
                dtype=np.float64
            )

            row["shape"] = str(cir.shape)
            row["dtype"] = str(cir.dtype)
            valid_shape = (
                len(cir.shape) == 3
                and cir.shape[0] == 3
                and cir.shape[2] == 368
            )

            row["valid_shape"] = valid_shape

            if len(cir.shape) != 3:
                row["error"] = "Invalid CIR dimensions"
                results.append(row)
                continue

            n_frames = cir.shape[1]

            row["frames"] = n_frames
            row["timestamp_length_ok"] = (
                len(t) == n_frames
            )

            row["timestamp_finite"] = bool(
                np.isfinite(t).all()
            )

            if len(t) >= 2:

                dt = np.diff(t)

                row["duplicate_or_backward_t"] = int(
                    (dt <= 0).sum()
                )

                row["large_time_gaps"] = int(
                    (dt > 0.03).sum()
                )

                row["median_dt"] = float(
                    np.median(dt)
                )

                row["duration_s"] = float(
                    t[-1] - t[0]
                )

                row["t_min"] = float(t[0])
                row["t_max"] = float(t[-1])

            else:

                row["duplicate_or_backward_t"] = 0
                row["large_time_gaps"] = 0
                row["median_dt"] = np.nan
                row["duration_s"] = 0.0
                row["t_min"] = np.nan
                row["t_max"] = np.nan

            row["radar_serials_present"] = (
                "radar_serials" in f
            )

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

                nan_count += int(
                    np.isnan(block).sum()
                )

                inf_count += int(
                    np.isinf(block).sum()
                )

                affected_frames += int(
                    np.any(
                        ~np.isfinite(block),
                        axis=(0, 2)
                    ).sum()
                )

                zero_frames += int(
                    np.all(
                        block == 0,
                        axis=(0, 2)
                    ).sum()
                )

            row["nan_count"] = nan_count
            row["inf_count"] = inf_count

            row["affected_frames"] = affected_frames
            row["all_zero_frames"] = zero_frames


            if (
                len(t) > 0
                and np.isfinite(t[0])
                and np.isfinite(t[-1])
            ):

                outside = (
                    (labels["t_start"] < t[0] - 0.02)
                    |
                    (labels["t_end"] > t[-1] + 0.02)
                )

                row["events_outside_stream"] = int(
                    outside.sum()
                )

            else:

                row["events_outside_stream"] = np.nan

    except Exception as exc:

        row["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    results.append(row)

    print(
        f"Finished {stream_id} | "
        f"Events: {row['event_count']} | "
        f"Error: {row['error'] or 'None'}",
        flush=True
    )

quality = pd.DataFrame(results)

quality.to_csv(
    REPORTS / "04_data_quality_per_stream.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([label_stats]).to_csv(
    REPORTS / "04_label_quality_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== HDF5 DATA QUALITY ==========")

columns = [
    "stream_id",
    "event_count",
    "valid_shape",
    "frames",
    "nan_count",
    "inf_count",
    "affected_frames",
    "duplicate_or_backward_t",
    "large_time_gaps",
    "events_outside_stream",
    "error",
]

print(
    quality.reindex(
        columns=columns
    ).to_string(index=False)
)


total_matched_events = int(
    quality["event_count"].sum()
)

expected_events = len(df)

print("\n========== EVENT MATCHING ==========")

print("Expected events:", expected_events)
print("Matched events:", total_matched_events)

expected_stream_ids = set(
    df["stream_id"].unique()
)

scanned_stream_ids = set(
    quality["stream_id"]
)

missing_streams = (
    expected_stream_ids - scanned_stream_ids
)

extra_streams = (
    scanned_stream_ids - expected_stream_ids
)

print("Missing streams:", sorted(missing_streams))
print("Extra streams:", sorted(extra_streams))

scan_errors = quality[
    quality["error"].fillna("").ne("")
]

if not scan_errors.empty:
    print("\nFiles with errors:")
    print(
        scan_errors[
            ["stream_id", "error"]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Some HDF5 files failed. "
        "Check 04_data_quality_per_stream.csv"
    )

if missing_streams:
    raise RuntimeError(
        f"Missing HDF5 streams: {sorted(missing_streams)}"
    )

if extra_streams:
    raise RuntimeError(
        f"Unexpected HDF5 streams: {sorted(extra_streams)}"
    )

if total_matched_events != expected_events:
    raise RuntimeError(
        "Event matching failed: "
        f"{total_matched_events}/{expected_events}"
    )


metric_cols = [
    "nan_count",
    "inf_count",
    "affected_frames",
    "all_zero_frames",
    "duplicate_or_backward_t",
    "large_time_gaps",
    "events_outside_stream",
]

totals = (
    quality.reindex(columns=metric_cols)
    .apply(pd.to_numeric, errors="coerce")
    .sum()
)

invalid_shape_count = int(
    (~quality["valid_shape"].eq(True)).sum()
)

bad_timestamp_length = int(
    (~quality["timestamp_length_ok"].eq(True)).sum()
)

bad_timestamp_values = int(
    (~quality["timestamp_finite"].eq(True)).sum()
)

print("\n========== FINAL SUMMARY ==========")

print("Files checked:", len(quality))
print("Events matched:", total_matched_events)
print("Invalid CIR shapes:", invalid_shape_count)
print("Invalid timestamp lengths:", bad_timestamp_length)
print("Non-finite timestamp streams:", bad_timestamp_values)
print("NaN:", int(totals["nan_count"]))
print("Inf:", int(totals["inf_count"]))
print("Affected frames:", int(totals["affected_frames"]))
print("All-zero frames:", int(totals["all_zero_frames"]))
print("Time gaps:", int(totals["large_time_gaps"]))
print(
    "Events outside stream:",
    int(totals["events_outside_stream"])
)


text = f"""# DATA QUALITY ANALYSIS

## 1. Label Quality

- Total fall events: {len(df)}
- Quality low: {int(low.sum())}
- Needs review: {int(review.sum())}
- Both flags: {int((low & review).sum())}
- At least one flag: {int(flagged.sum())}
- Ignore true: {int(ignore.sum())}
- Duplicate event IDs: {label_stats['duplicate_event_ids']}
- Invalid event intervals: {label_stats['invalid_intervals']}
- Missing CSV cells: {label_stats['missing_cells']}

## 2. HDF5 Quality

- Files scanned: {len(quality)}
- Events matched: {total_matched_events}/{expected_events}
- Invalid CIR shapes: {invalid_shape_count}
- Invalid timestamp lengths: {bad_timestamp_length}
- Non-finite timestamp streams: {bad_timestamp_values}
- NaN values: {int(totals['nan_count'])}
- Inf values: {int(totals['inf_count'])}
- Affected frames: {int(totals['affected_frames'])}
- All-zero frames: {int(totals['all_zero_frames'])}
- Duplicate/backward timestamp steps: {int(totals['duplicate_or_backward_t'])}
- Time gaps above 0.03s: {int(totals['large_time_gaps'])}
- Events outside stream bounds: {int(totals['events_outside_stream'])}

## 3. Data Quality Observations

- HDF5 structure, timestamps, and signal values were checked.
- All annotated events were matched with their HDF5 streams.
- Quality flags must be considered during model development.
- All-zero frames are not automatically invalid.
- Consistent timestamps do not guarantee perfect raw measurements.
- Low-quality annotations are not equivalent to missing values.

## 4. Recommendations

- Check input quality before preprocessing.
- Investigate low-quality and needs-review fall events.
- Respect the challenge evaluation rules for flagged events.
- Avoid learning artificial stream-boundary patterns.
"""

(REPORTS / "04_data_quality_report.md").write_text(
    text,
    encoding="utf-8"
)

print("\n================================")
print("DATA QUALITY ANALYSIS COMPLETED")
print("================================")

print("Results saved in:", REPORTS)
