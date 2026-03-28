"""Model evaluation entry point.

Responsibilities include loading models, running tests, computing attack metrics, and plotting results.
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Literal

from configs.config import get_config, Config
from dataloaders.load import TorchDatasetBundle, gen_torch_dataset
from models import SimpleCNN
from models.losses import get_loss_and_scoring
from utils.attack import calc_key_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@torch.inference_mode()
def test(
    model: torch.nn.Module,
    data: TorchDatasetBundle,
    cfg: Config,
) -> pd.DataFrame:
    """Run inference on the test set and compute attack evaluation metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # The test set uses only the attack split.
    test_loader = DataLoader(
        data["test_ds"],
        batch_size=cfg.test_batch_size,
        shuffle=False,
        pin_memory=True,
        # num_workers=2,
    )
    target_dtype = torch.float32 if cfg.num_classes == 1 else torch.long

    # Switch the model to evaluation mode and reuse the training scoring function.
    model = model.to(device).eval()
    criterion, scoring = get_loss_and_scoring(name=cfg.loss_type)

    # Run forward inference and accumulate the test loss.
    total_test_loss = 0.0
    test_batches = 0
    pred_list = []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device).to(target_dtype)
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
        total_test_loss += loss.item()
        test_batches += 1
        pred_list.append(outputs)

    avg_test_loss = total_test_loss / test_batches if test_batches else 0.0
    pred = torch.cat(pred_list, dim=0)
    metrics = calc_key_metrics(
        scores=scoring(pred).cpu().numpy(),
        median=data["median_test"],
        key=data["key_test"],
        num_traces_list=range(50, 5001, 50),
        repeat=cfg.repeat,
    )

    logger.info(f"Test Loss: {avg_test_loss:.6f}")
    return metrics


def run(
    model_path: Path,
    load_model: Literal["state_dict", "frozen"],
    config_file: Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load the model, generate data, call test, and return a metrics DataFrame."""
    cfg = get_config(config_file, **kwargs)
    # Build the full data bundle for test.
    data = gen_torch_dataset(cfg)

    # Load the model according to the export format.
    if load_model == "state_dict":
        model = SimpleCNN(num_classes=cfg.num_classes)
        state = torch.load(model_path)
        model.load_state_dict(state)
        model = model
        print(f"Model loaded from state_dict: {model_path}")
    elif load_model == "frozen":
        model = torch.jit.load(model_path)
        print(f"Model loaded from frozen file: {model_path}")
        try:
            model = torch.jit.freeze(model)
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown load_mode: {cfg.load_model}")

    # Run the test and return attack metrics.
    metrics = test(model=model, data=data, cfg=cfg)
    return metrics


def plot_metrics(
    metrics_file: Path,
    save_fig: bool = False,
    root_dir: Path = Path("outputs/attack"),
) -> list[tuple[plt.Figure, str]]:
    """Load CSV metrics and plot them, returning the created figure list."""
    """
    Plot attack metric charts: Top-1 SR, Top-3 SR, Guessing Entropy.
    """
    metrics = pd.read_csv(metrics_file)
    print(f"Metrics loaded from: {metrics_file}")

    traces = metrics["num_traces"]
    top1 = metrics["top1"]
    top3 = metrics["top3"]
    ge = metrics["median_rank"]

    # Plot configuration
    fig_configs = [
        {
            "data": top1,
            "ylabel": "Top-1 Success Rate",
            "title": "Top-1 Success Rate vs Number of Traces",
            "filename": "top1_sr",
            "ylim": (0, 1.05),
        },
        {
            "data": top3,
            "ylabel": "Top-3 Success Rate",
            "title": "Top-3 Success Rate vs Number of Traces",
            "filename": "top3_sr",
            "ylim": (0, 1.05),
        },
        {
            "data": ge,
            "ylabel": "Guessing Entropy",
            "title": "Guessing Entropy vs Number of Traces",
            "filename": "guessing_entropy",
            "ylim": (None),
        },
    ]

    figures = []

    for cfg in fig_configs:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            traces,
            cfg["data"],
            color="#5B8DB8",
            linewidth=1.5,
            linestyle=":",
        )

        ax.set_xlabel("Number of Traces", fontsize=12)
        ax.set_ylabel(cfg["ylabel"], fontsize=12)
        ax.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if cfg["ylim"]:
            ax.set_ylim(cfg["ylim"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        figures.append((fig, cfg["filename"]))

        if save_fig:
            root_dir.mkdir(parents=True, exist_ok=True)
            for fmt in ["png", "svg", "pdf"]:
                dpi = 300 if fmt == "png" else None
                fig.savefig(
                    root_dir / f"{cfg['filename']}.{fmt}",
                    format=fmt,
                    dpi=dpi,
                    bbox_inches="tight",
                )

    plt.show()

    return figures


if __name__ == "__main__":
    root_dir = Path("outputs")
    metrics_path = root_dir / "attack/attack_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # metrics = run(
    #     model_path=root_dir / "best_model.pth", config_file=root_dir / "config.toml"
    # )
    # metrics.to_csv(metrics_path, index=False)
    plot_metrics(metrics_path, save_fig=True)
