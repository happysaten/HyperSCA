"""Training entry point.

Responsibilities include building data loaders, running training and validation,
recording metrics, and returning the best weights.
"""

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import copy
from pathlib import Path
import logging
from sys import stdout

from configs.config import get_config, Config
from dataloaders.load import TorchDatasetBundle, gen_torch_dataset
from models import SimpleCNN
from models.losses import get_loss_and_scoring
from utils.attack import calc_key_metrics
from utils.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(stdout)
# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
logger.addHandler(handler)


def train(
    model: torch.nn.Module,
    data: TorchDatasetBundle,
    cfg: Config,
) -> tuple[float, dict[str, torch.Tensor] | None]:
    """Train the model and return the best SR together with its state dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Build the training and validation DataLoaders.
    train_loader = DataLoader(
        data["train_ds"],
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        data["val_ds"],
        batch_size=cfg.test_batch_size,
        shuffle=False,
        pin_memory=True,
        # num_workers=2,
    )
    # Target dtype differs for binary and multiclass tasks.
    target_dtype = torch.float32 if cfg.num_classes == 1 else torch.long

    # Move the model to the target device and prepare the loss, optimizer, and scheduler.
    model = model.to(device)
    criterion, scoring = get_loss_and_scoring(name=cfg.loss_type)
    optimizer = getattr(torch.optim, cfg.optimizer)(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Track epoch and step metrics during training.
    metrics_logger = MetricsLogger()

    # AMP configuration: enabled only when CUDA is available.
    use_amp = getattr(cfg, "use_amp", False) and device.type == "cuda"
    # cfg.amp_dtype supports "fp16" / "bfloat16"; the default is fp16.
    amp_dtype_cfg = getattr(cfg, "amp_dtype", "fp16")
    amp_dtype = torch.float16 if amp_dtype_cfg == "fp16" else torch.bfloat16
    if use_amp:
        scaler = GradScaler()

    # State variables for the training loop.
    best_sr = float("-inf")
    best_model = None
    global_step = 0
    log_interval = cfg.log_interval  # Configured value.
    # When log_interval > 0, record training loss by step.
    do_step_logging = isinstance(log_interval, int) and log_interval > 0

    # Run one batch first to confirm input and output shapes.
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device).to(target_dtype)
    if use_amp:
        with autocast(enabled=True, dtype=amp_dtype, device_type="cuda"):
            out = model(x)
    else:
        out = model(x)
    logger.info(f"x shape: {x.shape}, y shape: {y.shape}, output shape: {out.shape}")

    for epoch in range(1, cfg.epochs + 1):
        # Training phase.
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        # Used only when step logging is enabled.
        step_loss = 0.0
        step_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).to(target_dtype)
            optimizer.zero_grad()

            if use_amp:
                # AMP forward and backward pass.
                with autocast(enabled=True, dtype=amp_dtype, device_type="cuda"):
                    outputs = model(x)
                    loss = criterion(outputs.squeeze(-1), y)
                # Scale the loss before the backward update.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss = criterion(outputs.squeeze(-1), y)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1
            # Accumulate only when step logging is enabled.
            if do_step_logging:
                step_loss += loss.item()
                step_batches += 1

                # Log once every accumulated log_interval batches.
                if step_batches == log_interval:
                    global_step += 1
                    current_lr = optimizer.param_groups[0]["lr"]
                    avg_step_loss = step_loss / step_batches

                    metrics_logger.log(
                        step=global_step,
                        epoch=epoch,
                        train_loss=avg_step_loss,
                        val_loss=None,
                        val_sr=None,
                        lr=current_lr,
                    )
                    step_loss = 0.0
                    step_batches = 0

        # Handle the final partial batch group at the end of the epoch.
        if do_step_logging and step_batches > 0:
            global_step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            avg_step_loss = step_loss / step_batches

            metrics_logger.log(
                step=global_step,
                epoch=epoch,
                train_loss=avg_step_loss,
                val_loss=None,
                val_sr=None,
                lr=current_lr,
            )

        avg_train_loss = total_train_loss / train_batches if train_batches else 0.0

        # Validation phase.
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            pred_list = []
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).to(target_dtype)
                # Use AMP during validation as well.
                if use_amp:
                    with autocast(enabled=True, dtype=amp_dtype, device_type="cuda"):
                        outputs = model(x)
                        loss = criterion(outputs.squeeze(-1), y)
                else:
                    outputs = model(x)
                    loss = criterion(outputs.squeeze(-1), y)

                total_val_loss += loss.item()
                val_batches += 1
                pred_list.append(outputs)

            pred = torch.cat(pred_list, dim=0)
            sr = calc_key_metrics(
                scores=scoring(pred).cpu().float().numpy(),
                median=data["median_val"],
                key=data["key_val"],
                num_traces_list=[cfg.num_traces],
                repeat=cfg.repeat,
            )[cfg.metrics][0]

        avg_val_loss = total_val_loss / val_batches if val_batches else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        # Record validation metrics for each epoch.
        metrics_logger.log(
            step=global_step,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_sr=sr,
            lr=current_lr,
            is_epoch_end=True,
        )

        logger.info(
            f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | Val SR: {sr:.2f} | LR: {current_lr:.6f}"
        )

        if sr > best_sr:
            best_sr = sr
            if sr >= cfg.val_threshold:
                best_model = copy.deepcopy(model.state_dict())
            if sr >= 0.99:
                logger.info(f"Early stopping at epoch {epoch} with SR: {sr:.2f}")
                break
            # torch.save(model.state_dict(), f"{cfg.output_dir}/{cfg.model_file}")

        scheduler.step()

    # Reserved metric persistence logic, disabled by default.
    # metrics_logger.save(root_dir=cfg.output_dir, filename="training_metrics.csv")
    # cfg.save(filename="config.toml")
    logger.info(f"Latest Success Rate: {sr:.2f}, Best Success Rate: {best_sr:.2f}")

    return best_sr, best_model


def run(
    config_file: Path | None = None, **kwargs
) -> tuple[float, dict[str, torch.Tensor] | None]:
    """Load the config, build the data and model, and run training."""
    cfg = get_config(config_file, **kwargs)
    print(f"Config:\n{cfg.to_toml()}")
    model = SimpleCNN(num_classes=cfg.num_classes)
    # model = TimeSeriesTransformer_v2(num_classes=cfg.num_classes, seq_len=700)
    data = gen_torch_dataset(cfg)
    return train(
        model=model,
        data=data,
        cfg=cfg,
    )


if __name__ == "__main__":
    run(config_file=Path("configs/AES_RD.toml"))

    # # Automatically plot the loss curve after training
    # import subprocess
    # subprocess.run([
    #     "python", "loss_plot.py",
    #     "-i", "outputs/training_metrics.csv",
    #     "-o", "outputs/loss_curve.png"
    # ])
