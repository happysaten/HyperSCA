"""
Batch plotting utility: group CSVs by subdirectory, aggregate metrics across runs (mean±std), and draw subplots.
Usage: call plot_multi(path, output_name, max_cols) directly.
"""

import matplotlib
import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(filepath: str) -> pd.DataFrame:
    """Load a CSV and skip metadata rows that start with '# '."""
    with open(filepath, "r", encoding="utf-8") as f:
        skip_rows = 0
        for line in f:
            if line.startswith("# "):
                skip_rows += 1
            else:
                break
    return pd.read_csv(filepath, skiprows=skip_rows)


def parse_folder_value(folder_name: str) -> tuple:
    """
    Parse a folder name to extract the parameter and value (supports = / _ / -, including scientific notation).
    Returns (param_name, numeric_value or original value, original_name).
    """
    # Try matching param=value / param_value / param-value, including scientific notation.
    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*?)([=_-])(.+)$", folder_name)
    if match:
        param_name = match.group(1)
        separator = match.group(2)
        value_str = match.group(3)

        # If the separator is '_' or '-', make sure the suffix is a complete numeric value.
        if separator in ("_", "-"):
            # Try matching again to ensure the suffix is a complete numeric value.
            # Match scientific notation: digits(optional decimal)e/E(optional sign)digits.
            scientific_match = re.match(
                r"^([a-zA-Z_][a-zA-Z0-9_]*?)([=_-])(\d+\.?\d*[eE][+-]?\d+|\d+\.\d+|\d+)$",
                folder_name,
            )
            if scientific_match:
                param_name = scientific_match.group(1)
                value_str = scientific_match.group(3)
            elif not value_str[0].isdigit():
                # Non-digit prefixes are treated as unparseable and the original name is returned.
                return ("", folder_name, folder_name)

        try:
            # Try converting to a numeric value (scientific notation supported).
            # First check whether scientific notation markers or a decimal point are present.
            if "e" in value_str.lower() or "E" in value_str or "." in value_str:
                numeric_value = float(value_str)
            else:
                # Try integer conversion.
                try:
                    numeric_value = int(value_str)
                except ValueError:
                    # It may be a float without a decimal point (rare).
                    numeric_value = float(value_str)

            return (param_name, numeric_value, folder_name)
        except ValueError:
            # Fall back to string ordering if numeric conversion fails.
            return (param_name, value_str, folder_name)

    # Return the original name if no pattern matches.
    return ("", folder_name, folder_name)


def find_csv_files_grouped(path: Path) -> dict:
    """Find all CSVs in a directory, group by top-level parent, and return a dict sorted by parsed values."""
    grouped = defaultdict(list)

    if path.is_file() and path.suffix == ".csv":
        grouped[path.parent.name].append(path)
    elif path.is_dir():
        for csv_file in sorted(path.rglob("*.csv")):
            # Use the parent directory relative to input_path as the grouping key.
            rel_path = csv_file.relative_to(path)
            if len(rel_path.parts) > 1:
                group_key = rel_path.parts[0]  # First-level subdirectory.
            else:
                group_key = path.name  # Directly under input_path.
            grouped[group_key].append(csv_file)

    # Parse and sort the groups.
    sorted_groups = []
    for group_name, csv_files in grouped.items():
        param_name, numeric_value, original_name = parse_folder_value(group_name)
        sorted_groups.append((param_name, numeric_value, original_name, csv_files))

    # Sort by parameter name and value.
    sorted_groups.sort(key=lambda x: (x[0], x[1]))

    # Return the sorted dictionary.
    return {item[2]: item[3] for item in sorted_groups}


def aggregate_dataframes(dfs: list) -> dict[str, np.ndarray]:
    """Merge multiple DataFrames: align by step and compute mean/std; for a single file, return the values with std set to 0."""
    if len(dfs) == 1:
        df = dfs[0]
        steps = df["step"].values
        train_losses = df["train_loss"].values

        epoch_df = df[df["is_epoch_end"]].copy()
        epoch_steps = epoch_df["step"].values
        val_losses = epoch_df["val_loss"].values
        val_srs = epoch_df["val_success_rate"].values

        return {
            "steps": steps,
            "train_mean": train_losses,
            "train_std": np.zeros_like(train_losses),
            "epoch_steps": epoch_steps,
            "val_mean": val_losses,
            "val_std": np.zeros_like(val_losses),
            "sr_mean": val_srs,
            "sr_std": np.zeros_like(val_srs),
        }

    # Multiple files: align to the minimum length before computing statistics.
    min_len = min(len(df) for df in dfs)

    train_all = np.array([df["train_loss"].values[:min_len] for df in dfs])
    steps = dfs[0]["step"].values[:min_len]

    # Epoch data.
    epoch_dfs = [df[df["is_epoch_end"]].copy() for df in dfs]
    min_epoch_len = min(len(edf) for edf in epoch_dfs)

    val_all = np.array([edf["val_loss"].values[:min_epoch_len] for edf in epoch_dfs])
    sr_all = np.array(
        [edf["val_success_rate"].values[:min_epoch_len] for edf in epoch_dfs]
    )
    epoch_steps = epoch_dfs[0]["step"].values[:min_epoch_len]

    return {
        "steps": steps,
        "train_mean": np.mean(train_all, axis=0),
        "train_std": np.std(train_all, axis=0),
        "epoch_steps": epoch_steps,
        "val_mean": np.mean(val_all, axis=0),
        "val_std": np.std(val_all, axis=0),
        "sr_mean": np.mean(sr_all, axis=0),
        "sr_std": np.std(sr_all, axis=0),
    }


def plot_single_ax(ax, data: dict, title: str = "", n_runs: int = 1):
    """Draw Train Loss / Val Loss / Val SR on a single subplot (mean + ±1 std band)."""
    steps = data["steps"]
    train_mean, train_std = data["train_mean"], data["train_std"]
    epoch_steps = data["epoch_steps"]
    val_mean, val_std = data["val_mean"], data["val_std"]
    sr_mean, sr_std = data["sr_mean"], data["sr_std"]

    # Soft, low-saturation colors.
    color_train = "#5B8DB8"
    color_val = "#E8A87C"
    color_sr = "#7CAE7A"

    # Left axis shows loss, right axis shows success rate (fixed 0~1).
    ax.plot(
        steps,
        train_mean,
        linestyle="-",
        linewidth=0.8,
        color=color_train,
        alpha=0.7,
        label="Train Loss",
    )
    if n_runs > 1:
        ax.fill_between(
            steps,
            train_mean - train_std,
            train_mean + train_std,
            color=color_train,
            alpha=0.2,
        )

    ax.plot(
        epoch_steps,
        val_mean,
        linestyle="--",
        linewidth=1.5,
        color=color_val,
        marker="o",
        markersize=3,
        label="Val Loss",
    )
    if n_runs > 1:
        ax.fill_between(
            epoch_steps,
            val_mean - val_std,
            val_mean + val_std,
            color=color_val,
            alpha=0.2,
        )

    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9, color=color_train)
    ax.tick_params(axis="y", labelcolor=color_train, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Include run count in the title.
    title_suffix = f" (n={n_runs})" if n_runs > 1 else ""
    ax.set_title(f"{title}{title_suffix}", fontsize=10, fontweight="bold", pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, color="#aaaaaa")

    # Right axis: success rate (fixed 0~1).
    ax2 = ax.twinx()
    ax2.plot(
        epoch_steps,
        sr_mean,
        linestyle=":",
        linewidth=1.5,
        color=color_sr,
        marker="s",
        markersize=3,
        label="Val SR",
    )
    if n_runs > 1:
        ax2.fill_between(
            epoch_steps,
            np.clip(sr_mean - sr_std, 0, 1),
            np.clip(sr_mean + sr_std, 0, 1),
            color=color_sr,
            alpha=0.2,
        )

    ax2.set_ylabel("SR", fontsize=9, color=color_sr)
    ax2.tick_params(axis="y", labelcolor=color_sr, labelsize=8)
    ax2.set_ylim(0, 1)  # Fix the right-axis range.
    ax2.grid(False)

    return ax, ax2


def plot_multi(
    path: Path, output_name: str = "multi_loss_curve.png", max_cols: int = 3
):
    """Draw subplots for each group (auto layout) and save to path/output_name."""
    grouped_files = find_csv_files_grouped(path)

    if not grouped_files:
        print(f"No CSV files found in {path}")
        return

    n = len(grouped_files)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for idx, (group_name, csv_files) in enumerate(grouped_files.items()):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        try:
            dfs = [load_metrics(str(f)) for f in csv_files]
            data = aggregate_dataframes(dfs)

            # Parse the folder name to get a friendlier title.
            param_name, numeric_value, _ = parse_folder_value(group_name)
            if param_name:
                # Format numeric values.
                if isinstance(numeric_value, float):
                    if abs(numeric_value) < 0.01:
                        title = f"{param_name}={numeric_value:.2e}"
                    else:
                        title = f"{param_name}={numeric_value:.4g}"
                else:
                    title = f"{param_name}={numeric_value}"
            else:
                title = group_name

            plot_single_ax(ax, data, title=title, n_runs=len(dfs))
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error:\n{e}",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_title(group_name, fontsize=10)

    # Hide extra subplots and add a shared legend.
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    # Add a shared legend.
    handles, labels = [], []
    color_train, color_val, color_sr = "#5B8DB8", "#E8A87C", "#7CAE7A"
    handles.append(plt.Line2D([0], [0], color=color_train, linewidth=1, alpha=0.7))
    labels.append("Train Loss")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color=color_val,
            linewidth=1.5,
            linestyle="--",
            marker="o",
            markersize=3,
        )
    )
    labels.append("Val Loss")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color=color_sr,
            linewidth=1.5,
            linestyle=":",
            marker="s",
            markersize=3,
        )
    )
    labels.append("Val SR")
    # Std band legend.
    handles.append(plt.fill_between([], [], [], color="gray", alpha=0.2))
    labels.append("±1 Std")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save and close the figure.
    output_path = path / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-plot to {output_path}")


if __name__ == "__main__":
    plot_multi(
        path=Path("outputs/sweep/optimizer=SGD"),
        output_name="multi_loss_curve.png",
        max_cols=4,
    )
