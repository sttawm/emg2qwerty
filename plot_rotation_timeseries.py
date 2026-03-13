"""Plot rotation weights per mini-batch step (time-ordered, shuffle=False)."""

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
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

OFFSETS = [-2, -1, 0, 1, 2]
BANDS = ["left", "right"]
MAX_STEPS = 127  # one epoch


def offset_tag(band: str, offset: int) -> str:
    sign = "neg" if offset < 0 else "pos"
    return f"rotation_step/{band}/offset_{sign}{abs(offset)}"


def find_local_event_dir() -> Path:
    logs_root = Path("logs")
    candidates = sorted(
        logs_root.rglob("lightning_logs/version_0"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No local lightning_logs found under logs/")
    event_dir = candidates[0]
    print(f"Using local logs: {event_dir}")
    return event_dir


def load_scalar(ea: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = [e for e in ea.Scalars(tag) if e.step <= MAX_STEPS]
    return [e.step for e in events], [e.value for e in events]


def boundary_jump_metric(
    ea: EventAccumulator,
    band: str,
) -> None:
    """Print mean absolute change at session boundaries vs. within sessions."""
    steps_s, values_s = load_scalar(ea, "rotation_step/session")
    if not steps_s:
        print(f"  [{band}] No session data — skipping jump metric")
        return

    # Build a set of steps where a session boundary occurred
    boundary_steps: set[int] = set()
    prev_session = values_s[0]
    for step, val in zip(steps_s, values_s):
        if val != prev_session:
            boundary_steps.add(step)
            prev_session = val

    within_changes: list[float] = []
    boundary_changes: list[float] = []

    for offset in OFFSETS:
        tag = offset_tag(band, offset)
        steps, values = load_scalar(ea, tag)
        if len(steps) < 2:
            continue
        for j in range(1, len(steps)):
            delta = abs(values[j] - values[j - 1])
            if steps[j] in boundary_steps:
                boundary_changes.append(delta)
            else:
                within_changes.append(delta)

    mean_within = np.mean(within_changes) if within_changes else float("nan")
    mean_boundary = np.mean(boundary_changes) if boundary_changes else float("nan")
    ratio = mean_boundary / mean_within if mean_within > 0 else float("nan")
    print(
        f"  [{band}] mean |Δw| within={mean_within:.4f}  "
        f"boundary={mean_boundary:.4f}  ratio={ratio:.2f}x  "
        f"(n_within={len(within_changes)}, n_boundary={len(boundary_changes)})"
    )


def main():
    event_dir = find_local_event_dir()
    print("Loading TensorBoard events...")
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    color_cycle = plt.cm.tab10.colors

    for ax, band in zip(axes, BANDS):
        for i, offset in enumerate(OFFSETS):
            tag = offset_tag(band, offset)
            steps, values = load_scalar(ea, tag)
            if steps:
                ax.plot(steps, values, label=f"offset {offset:+d}",
                        color=color_cycle[i], linewidth=1.2, alpha=0.85)
            else:
                print(f"  No data: {tag}")

        # Overlay session boundaries
        steps_s, values_s = load_scalar(ea, "rotation_step/session")
        if steps_s:
            prev = values_s[0]
            for step, val in zip(steps_s, values_s):
                if val != prev:
                    ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
                    prev = val

        ax.set_title(f"{band.capitalize()} band")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Mini-batch step (time-ordered)")
    fig.suptitle("Rotation weights over time (shuffle=False)", fontsize=14)
    fig.tight_layout()
    out = Path("rotation_timeseries.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    print("\nSession boundary jump metric (mean |Δw| per step):")
    for band in BANDS:
        boundary_jump_metric(ea, band)


if __name__ == "__main__":
    main()
