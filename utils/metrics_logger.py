import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Training metrics logger. Records key step/epoch/train/val metrics and supports CSV export."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def log(
        self,
        step: int | None = None,
        epoch: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        val_sr: float | None = None,
        lr: float | None = None,
        is_epoch_end: bool = False,
    ) -> None:
        """Record one training/validation metrics entry. All numeric arguments may be None."""
        self.records.append(
            {
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loss is not None else "",
                "val_success_rate": val_sr if val_sr is not None else "",
                "learning_rate": lr,
                "is_epoch_end": is_epoch_end,
            }
        )

    def save(self, root_dir: Path | str, filename: str | None = None) -> Path:
        """Save the records as CSV. If filename is omitted, use a timestamped name and return the file path."""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"

        filepath = root_dir / filename

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            # Write CSV data
            if self.records:
                writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)

        print(f"Metrics saved to: {filepath}")
        return filepath
