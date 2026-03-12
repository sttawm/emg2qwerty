"""Plot rotation weight distributions (last epoch) as bar charts, per band."""

import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

GCS_PATH = "gs://emg2qwerty-team-logs/logs/learned_soft_rotation_per_band_v2/lightning_logs/version_0/"
OFFSETS = [-2, -1, 0, 1, 2]
BANDS = ["left", "right"]


def download_events(tmp_dir: Path) -> Path:
    local_dir = tmp_dir / "events"
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", GCS_PATH, str(local_dir)],
        check=True,
    )
    return local_dir


def get_last_histogram(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (bin_centers, counts) for the last epoch of a histogram tag."""
    if tag not in ea.Tags().get("histograms", []):
        print(f"  Tag not found: {tag}")
        return None
    events = ea.Histograms(tag)
    if not events:
        return None
    last = events[-1]  # last epoch
    h = last.histogram_value
    # TensorBoard histogram: left_edges + right_edges + counts
    edges = list(h.left_edge) + [h.right_edge[-1]]
    counts = np.array(h.right_edge)  # counts stored in right_edge field
    # Actually use the standard fields
    counts = np.array(h.bucket)
    edges = np.array(h.bucket_limit)
    # Compute bin centers from edges
    bin_centers = (edges[:-1] + edges[1:]) / 2 if len(edges) > len(counts) else edges
    return bin_centers, counts


def get_last_histogram_v2(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (bin_centers, counts) for the last epoch."""
    if tag not in ea.Tags().get("histograms", []):
        print(f"  Tag not found: {tag}")
        return None
    events = ea.Histograms(tag)
    if not events:
        return None
    last = events[-1]
    h = last.histogram_value
    counts = np.array(h.bucket)
    limits = np.array(h.bucket_limit)
    # limits are right edges; prepend min as left edge
    left = np.concatenate([[h.min], limits[:-1]])
    right = limits
    centers = (left + right) / 2
    return centers, counts


BAND_COLORS = {"left": "darkorange", "right": "mediumpurple"}


def plot_band(ax_row, ea: EventAccumulator, band: str) -> None:
    color = BAND_COLORS[band]
    for ax, offset in zip(ax_row, OFFSETS):
        sign = "-" if offset < 0 else "_"
        tag = f"rotation/{band}/offset_{sign}{abs(offset)}"
        result = get_last_histogram_v2(ea, tag)
        if result is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            centers, counts = result
            probs = counts / counts.sum()
            ax.bar(centers, probs, width=0.05, color=color, edgecolor="none")
        ax.set_xlabel("Weight")
        ax.set_title(f"offset {offset:+d}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if offset != -2:
            ax.set_ylabel("")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("Downloading event files...")
        event_dir = download_events(tmp_dir)

        print("Loading TensorBoard events...")
        ea = EventAccumulator(str(event_dir), size_guidance={"histograms": 0})
        ea.Reload()

        print("Available histogram tags:")
        for tag in ea.Tags().get("histograms", []):
            print(f"  {tag}")

        fig, all_axes = plt.subplots(2, 5, figsize=(14, 7), sharey=True)
        fig.subplots_adjust(hspace=0.55)
        for ax_row, band in zip(all_axes, BANDS):
            plot_band(ax_row, ea, band)
            ax_row[0].set_ylabel(f"{band.capitalize()} band\nProbability", fontweight="bold")
        fig.suptitle("Rotation weight distributions (last epoch)", fontsize=14)
        fig.tight_layout()
        out = Path("rotation_histograms.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
