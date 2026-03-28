"""Experiment configuration definitions and TOML read/write helpers."""

import tomllib
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Literal, Any


@dataclass
class Config:
    """Centralized config object shared by training, testing, and tuning."""

    # -------------------- Data config --------------------
    # dataset_type: str = "ASCAD_v2"
    # dataset_path: str = "Datasets/ASCAD_v2/ascadv2-extracted.h5"
    # target_byte: int = 0
    dataset_type: str = "ASCAD_F"
    dataset_path: str = "../Datasets/ASCAD_F/ASCAD.h5"
    target_byte: int = 2
    # dataset_type: str = "AES_HD"
    # dataset_path: str = "Datasets/AES_HD/AES_HD.h5"
    # target_byte: int = 7
    prf_num: int = 35000
    atk_num: int = 5000
    val_size: int | float = 5000
    val_source: str = "profiling"  # "profiling" or "attack"
    standardize: bool = True
    random_select: bool = True
    random_state: int = 42

    # -------------------- Model config --------------------
    model_file: str = "best_model.pth"
    load_model: str = "state_dict"  # "state_dict" or "frozen"

    # -------------------- Train config --------------------
    batch_size: int = 128
    test_batch_size: int = 256
    optimizer: str = "AdamW"
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    loss_type: Literal["ID_CE", "HW_CE", "ID_DC", "HW_MSE", "HW_BD"] = "ID_DC"
    # loss_type: str = "hw_bd"
    # loss_type: str = "cross_entropy"
    log_interval: int = -1  # Log once every N batches

    use_amp: bool = False  # Whether to use gradient scaling for stability
    amp_dtype: str = "fp16"  # "float16", "bfloat16"

    # -------------------- Attack config --------------------
    num_traces: int = 1000
    repeat: int = 100
    metrics: str = "top3"

    # -------------------- Optuna config --------------------
    # storage: str = "sqlite:///optuna_study.db"
    # sampler_type: str = "TPE"
    # pruner_type: str = "Median"
    # timeout: int = 1800  # seconds, maximum duration per trial
    val_threshold: float = 0.15
    attack_ge_threshold: int = 32
    attack_ntge_threshold: int = 500

    # -------------------- Misc --------------------
    output_dir: str = "outputs"

    @property
    def num_classes(self) -> int:
        """Infer the output class count from the loss type."""
        match self.loss_type:
            case "ID_CE" | "ID_DC":
                return 256
            case "HW_CE":
                return 9
            case "HW_MSE" | "HW_BD":
                return 1
            case _:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

    @property
    def is_hw(self) -> bool:
        """Whether Hamming-weight labels are used."""
        match self.loss_type:
            case "ID_CE" | "ID_DC":
                return False
            case "HW_CE" | "HW_MSE" | "HW_BD":
                return True
            case _:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain dictionary."""
        return asdict(self)

    def to_toml(self) -> str:
        """Export as a TOML string."""
        import tomli_w

        return tomli_w.dumps(self.to_dict())

    def save(self, filename: str | None = None, root_dir: Path | None = None) -> Path:
        """Save the config to a TOML file and return its path."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"config_{timestamp}.toml"
        if root_dir is None:
            root_dir = Path(self.output_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        file_path = root_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_toml())
        print(f"Config saved to: {file_path}")
        return file_path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Construct a Config from a dictionary."""
        return cls(**d)

    @classmethod
    def from_toml_file(cls, filepath: Path | str, **kwargs: Any) -> "Config":
        """Load config from a TOML file and allow additional overrides."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            print(f"Loading config from: {filepath}")
            d = tomllib.load(f)
        d.update(kwargs)  # Override file contents with explicit arguments.
        return cls.from_dict(d)

    def override(self, **kwargs: Any) -> "Config":
        """Override existing fields in place and return self."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Key not found in config: {key}; cannot set value: {value}")
        return self


def get_config(config_file: Path | str | None = None, **kwargs: Any) -> Config:
    """Get a config: optionally read a file first, then override with explicit args."""
    if config_file:
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_file}")
        else:
            return Config.from_toml_file(config_file, **kwargs)
    else:
        return Config(**kwargs)


if __name__ == "__main__":
    # Test config loading and saving
    config = get_config(
        config_file="/root/shared-nvme/HyperSCA/configs/AES_HD.toml",
        batch_size=64,
        loss_type="HW_BD",
    )
    print(config.to_toml())
    print(config.num_classes)
