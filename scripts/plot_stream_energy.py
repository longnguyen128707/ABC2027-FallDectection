"""Plot stream CIR energy with ground-truth fall intervals."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK_FRAMES = 2_000
PLOT_STEP = 10


def load_events(path: Path) -> dict[str, list[tuple[float, float]]]:
    events: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stream_id", "t_start", "t_end"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            events[row["stream_id"]].append((float(row["t_start"]), float(row["t_end"])))
    return events


def stream_energy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        cir = handle["cir"]
        times = handle["t"][:]
        if cir.ndim != 3 or len(times) != cir.shape[1]:
            raise ValueError(f"{path}: expected cir shaped (radars, time, bins) and matching t")

        energy = np.empty(cir.shape[1], dtype=np.float64)
        for start in range(0, cir.shape[1], CHUNK_FRAMES):
            stop = min(start + CHUNK_FRAMES, cir.shape[1])
            frames = cir[:, start:stop, :].astype(np.float64)
            energy[start:stop] = np.mean(np.sum(frames**2, axis=2), axis=0)
    return times, energy


def plot_stream(
    path: Path,
    stream_id: str,
    events: list[tuple[float, float]],
    output_dir: Path,
) -> Path:
    times, energy = stream_energy(path)
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.plot(times[::PLOT_STEP], energy[::PLOT_STEP], color="black", linewidth=0.65, label="CIR energy")

    for index, (start, end) in enumerate(events):
        axis.axvspan(
            start,
            end,
            color="red",
            alpha=0.25,
            linewidth=0,
            label="Ground-truth fall interval" if index == 0 else None,
        )

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean radar energy (mean sum of CIR²)")
    axis.set_title(f"{stream_id}: CIR energy and ground-truth falls (n={len(events)})")
    axis.grid(True, alpha=0.25)
    if events:
        axis.legend()
    figure.tight_layout()

    output_path = output_dir / f"{stream_id}_energy_ground_truth.png"
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "stream_energy_ground_truth",
        help="Directory for the generated PNG plots.",
    )
    args = parser.parse_args()

    events_by_stream = load_events(ROOT / "streams_events.csv")
    stream_paths = sorted((ROOT / "streams").glob("*/*.h5"))
    if not stream_paths:
        raise FileNotFoundError(f"No stream HDF5 files found under {ROOT / 'streams'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in stream_paths:
        stream_id = f"{path.parent.name}_{path.stem.rsplit('_', 1)[-1]}"
        output_path = plot_stream(
            path,
            stream_id,
            events_by_stream.get(stream_id, []),
            args.output_dir,
        )
        print(f"{stream_id}: {len(events_by_stream.get(stream_id, []))} events -> {output_path}")


if __name__ == "__main__":
    main()
