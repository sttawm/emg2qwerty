"""Plot val/CER and val/loss training curves from GCS TensorBoard logs."""

import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

GCS_LOGS = "gs://emg2qwerty-team-logs/logs"

RUNS = {
    "Baseline (TDSConv)": "baseline_70",
    "No Rotation": "no_rotation",
    "Learned Rotation": "learned_soft_rotation_per_band_v2",
    "GRU 1x256 + Dropout": "gru_1x256_dropout_30",
    "LSTM 1x256": "lstm_1x256",
    "LSTM 1x256 + Dropout": "lstm_1x256_dropout_30",
    "LSTM 2x256": "lstm_2x256",
    "LSTM Bidir 1x256": "lstm_bidirectional_1x256+cosine_lr+no_dropout",
}

METRICS = ["val/CER", "val/loss"]
YLIMS = {"val/CER": (15, 30), "val/loss": (0.6, 1.5)}


def download_events(run_name: str, tmp_dir: Path) -> Path:
    local_dir = tmp_dir / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    gcs_path = f"{GCS_LOGS}/{run_name}/lightning_logs/"
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(local_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: gsutil failed for {run_name}: {result.stderr.strip()}")
    return local_dir


def load_scalar(event_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Walk event_dir recursively to find TensorBoard event files and extract tag."""
    event_files = list(event_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return [], []

    steps, values = [], []
    for ef in event_files:
        ea = EventAccumulator(str(ef.parent))
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for s in ea.Scalars(tag):
                steps.append(s.step)
                values.append(s.value)

    if not steps:
        return [], []

    # Sort by step and deduplicate
    paired = sorted(set(zip(steps, values)))
    steps, values = zip(*paired)
    return list(steps), list(values)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("Downloading TensorBoard event files from GCS...")
        data = {}
        for label, run_name in RUNS.items():
            print(f"  {label} ({run_name})")
            event_dir = download_events(run_name, tmp_dir)
            data[label] = {
                metric: load_scalar(event_dir, metric) for metric in METRICS
            }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        titles = {"val/CER": "Validation CER", "val/loss": "Validation Loss"}

        for ax, metric in zip(axes, METRICS):
            for label, run_data in data.items():
                steps, values = run_data[metric]
                if steps:
                    epochs = [s // 120 for s in steps]  # ~120 steps/epoch
                    ax.plot(epochs, values, label=label, linewidth=1.5)
                else:
                    print(f"  No data for {label} / {metric}")

            ax.set_title(titles[metric])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.split("/")[1])
            ax.set_ylim(*YLIMS[metric])
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out = Path("training_curves.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
