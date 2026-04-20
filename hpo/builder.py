"""Optuna hyperparameter sampling and model-building helpers.
Provides parameter sampling (sample_hp) and model construction (build_model) interfaces for Trainer/CNN/Transformer/MLP.
Note: this module only handles configuration and model validation; training logic is executed externally.
"""

from optuna.trial import Trial
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

from ._param import CategoricalParam, IntParam, FloatParam
from models import TimeSeriesTransformer_v2


# ============= Trainer search space sampling =============
class TrainerHyperBuilder:
    def sample_hp(self, trial: Trial) -> dict[str, Any]:
        """Sample Trainer-related hyperparameters and resolve dependencies.
        Returns a structured Trainer config dict (optimizer, lr, batch_size, epochs, amp, etc.).
        """

        hp = {}

        # ===== Step 1: sample optimizer parameters =====
        hp["optimizer"] = CategoricalParam(
            ["AdamW", "SGD", "RMSprop", "Adam", "Adadelta"]
        ).sample(trial, "optimizer_name")
        hp["lr"] = FloatParam(1e-6, 1e-2, log=True).sample(trial, "lr")

        # ===== Step 2: sample training parameters =====
        hp["batch_size"] = IntParam(32, 256, step=32).sample(trial, "batch_size")
        hp["epochs"] = IntParam(10, 50, step=5).sample(trial, "epochs")

        # ===== Step 3: sample AMP parameters =====
        hp["use_amp"] = CategoricalParam([True, False]).sample(trial, "use_amp")
        if hp["use_amp"]:
            hp["amp_dtype"] = CategoricalParam(["fp16", "bfloat16"]).sample(
                trial, "amp_dtype"
            )
        else:
            hp["amp_dtype"] = "fp16"  # default value
        # ===== Return structured config =====
        return hp


class _BaseHyperBuilder(ABC):
    @abstractmethod
    def sample_hp(self, trial: Trial) -> dict[str, Any]:
        """Sample model architecture and hyperparameters, returning a structured dict."""
        ...

    @abstractmethod
    def build_model(
        self, hp: dict[str, Any], input_shape: tuple[int, ...], output_dim: int
    ):
        """Build and return a model instance from the configuration."""
        ...

    @staticmethod
    def validate_model(model: nn.Module, input_shape: tuple[int, ...], output_dim: int):
        """Forward check: run a zero tensor through the model to ensure the output dimension matches expectations."""
        try:
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros((3, *input_shape))
                out = model(dummy)
            if out.shape[1] != output_dim:
                raise ValueError(
                    f"Model output dimension {out.shape[1]} does not match expected {output_dim}"
                )
        except Exception as e:
            raise ValueError(f"Model construction or forward validation failed: {e}") from e


class MLPHyperBuilder(_BaseHyperBuilder):
    def sample_hp(self, trial: Trial) -> dict[str, Any]:
        """Sample MLP config: layer count, per-layer units, activation, and dropout.
        Returns a structured dict for build_model.
        """
        # ===== Base parameters =====
        num_dense_layers = IntParam(2, 6).sample(trial, "mlp_num_dense_layers")
        activation = CategoricalParam(
            ["ReLU", "Tanh", "SELU", "ELU", "LeakyReLU"]
        ).sample(trial, "activation")

        # ===== Units per layer =====
        dense_layers = []
        for i in range(num_dense_layers):
            units = IntParam(256, 4096, step=64).sample(trial, f"mlp_dense_{i}_units")
            dense_layers.append({"units": units})

        # ===== Dropout parameters =====
        dropout_enabled = CategoricalParam(["On", "Off"]).sample(trial, "mlp_dropout")
        dropout = {"enabled": dropout_enabled == "On"}
        if dropout["enabled"]:
            dropout["rate"] = FloatParam(0.1, 0.5, step=0.1).sample(
                trial, "mlp_dropout_rate"
            )
        else:
            dropout["rate"] = None

        return {
            "dense_layers": dense_layers,
            "activation": activation,
            "dropout": dropout,
        }

    def build_model(
        self, hp: dict[str, Any], input_shape: tuple[int, ...], output_dim: int
    ):
        """Build the MLP from the config (Flatten first, then several LazyLinear + activation + optional Dropout).
        Returns a compiled model after forward validation.
        """
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must be (C, L), but got: {input_shape}")

        import torch.nn as nn

        layers = []
        # Flatten the input first.
        layers.append(nn.Flatten())
        activation = getattr(nn, hp["activation"])

        # Hidden layers
        for i, dense_cfg in enumerate(hp["dense_layers"]):
            layers.append(nn.LazyLinear(dense_cfg["units"]))
            layers.append(activation())
            # Add Dropout on non-final layers if enabled.
            if hp["dropout"]["enabled"] and i < len(hp["dense_layers"]) - 1:
                layers.append(nn.Dropout(p=hp["dropout"]["rate"]))

        # Output layer
        layers.append(nn.LazyLinear(output_dim))

        model = nn.Sequential(*layers)
        model.compile()

        # Validate the model.
        self.validate_model(model, input_shape, output_dim)

        return model


class CNNHyperBuilder(_BaseHyperBuilder):
    def sample_hp(self, trial: Trial) -> dict[str, Any]:
        """
        Sample CNN architecture and hyperparameters, returning a structured dict (conv_layers, dense_layers, pooling, dropout, activation).
        Sampling proceeds in logical steps: base -> convolution layers -> pooling -> dense -> dropout.
        """

        # ===== Step 1: sample base architecture parameters =====
        num_conv_layers = IntParam(1, 4).sample(trial, "num_conv_layers")
        num_dense_layers = IntParam(1, 3).sample(trial, "num_dense_layers")
        pooling_enabled = CategoricalParam(["On", "Off"]).sample(trial, "pooling")
        dropout_enabled = CategoricalParam(["On", "Off"]).sample(trial, "dropout")
        activation = CategoricalParam(
            ["ReLU", "Tanh", "SELU", "ELU", "LeakyReLU"]
        ).sample(trial, "activation")

        # ===== Step 2: sample convolution layer parameters =====
        conv_layers = []

        # First convolution layer
        first_filters = IntParam(4, 32, step=2).sample(trial, "conv_0_filters")
        first_conv = {
            "filters": first_filters,
            "kernel_size": CategoricalParam([3, 5, 7, 9]).sample(
                trial, "conv_0_kernel_size"
            ),
            "stride": CategoricalParam([2, 3, 4]).sample(trial, "conv_0_stride"),
            "groups": 1,
            "pointwise": False,
        }
        conv_layers.append(first_conv)

        # Subsequent convolution layers
        filters = first_filters
        for i in range(1, num_conv_layers):
            filters *= 2  # Double each layer.
            layer = {
                "filters": filters,
                "kernel_size": CategoricalParam([3, 5]).sample(
                    trial, f"conv_{i}_kernel_size"
                ),
                "stride": CategoricalParam([1]).sample(trial, f"conv_{i}_stride"),
                "groups": CategoricalParam([1, 2]).sample(trial, f"conv_{i}_groups"),
                "pointwise": CategoricalParam(["On", "Off"]).sample(
                    trial, f"conv_{i}_pointwise"
                )
                == "On",
            }
            conv_layers.append(layer)

        # ===== Step 3: sample pooling parameters =====
        pooling = {"enabled": pooling_enabled == "On"}
        if pooling["enabled"]:
            pooling["type"] = CategoricalParam(["Max", "Avg"]).sample(
                trial, "pooling_type"
            )
            pooling["kernel_size"] = CategoricalParam([2, 3, 4]).sample(
                trial, "pooling_kernel_size"
            )
        else:
            pooling["type"] = None
            pooling["kernel_size"] = None

        # ===== Step 4: sample fully connected layer parameters =====
        dense_layers = []
        for i in range(num_dense_layers):
            units = IntParam(256, 1024, step=64).sample(trial, f"dense_{i}_units")
            dense_layers.append({"units": units})

        # ===== Step 5: sample dropout parameters =====
        dropout = {"enabled": dropout_enabled == "On"}
        if dropout["enabled"]:
            dropout["rate"] = FloatParam(0.1, 0.5, step=0.1).sample(
                trial, "dropout_rate"
            )
        else:
            dropout["rate"] = None

        # ===== Return structured config =====
        return {
            "conv_layers": conv_layers,
            "activation": activation,
            "pooling": pooling,
            "dense_layers": dense_layers,
            "dropout": dropout,
        }

    def build_model(
        self, hp: dict[str, Any], input_shape: tuple[int, ...], output_dim: int
    ):
        """
        Build a 1D-CNN from hp (accepting input_shape=(C,L)); the final layer outputs output_dim.
        Return a compiled (torch.compile) nn.Sequential model and perform forward validation.
        """
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must be (C, L), but got: {input_shape}")
        in_channels, _ = input_shape

        import torch.nn as nn

        layers = []
        activation = getattr(nn, hp["activation"])  # Convert to an nn.Module class.

        # Build convolution layers.
        for _, conv_cfg in enumerate(hp["conv_layers"]):
            out_channels = conv_cfg["filters"]

            # Main convolution layer.
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_cfg["kernel_size"],
                    stride=conv_cfg["stride"],
                    padding=conv_cfg["kernel_size"] // 2,
                    groups=conv_cfg["groups"],
                )
            )
            layers.append(activation())

            # Pointwise convolution (1x1 conv)
            if conv_cfg["pointwise"]:
                layers.append(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                layers.append(activation())

            in_channels = out_channels

        # Add pooling.
        if hp["pooling"]["enabled"]:
            pool_type = hp["pooling"]["type"]
            kernel_size = hp["pooling"]["kernel_size"]

            if pool_type == "Max":
                layers.append(nn.MaxPool1d(kernel_size=kernel_size))
            elif pool_type == "Avg":
                layers.append(nn.AvgPool1d(kernel_size=kernel_size))

        # Flatten
        layers.append(nn.Flatten())

        # Build fully connected layers.
        for i, dense_cfg in enumerate(hp["dense_layers"]):
            layers.append(nn.LazyLinear(dense_cfg["units"]))
            layers.append(activation())

            # Add Dropout (not on the final layer).
            if hp["dropout"]["enabled"] and i < len(hp["dense_layers"]) - 1:
                layers.append(nn.Dropout(p=hp["dropout"]["rate"]))

        # Output layer (assume 10 classes).
        layers.append(nn.LazyLinear(output_dim))

        model = nn.Sequential(*layers)
        model.compile()

        self.validate_model(model, input_shape, output_dim)

        return model


class TransformerHyperBuilder(_BaseHyperBuilder):
    def sample_hp(self, trial: Trial) -> dict[str, Any]:
        """
        Sample Transformer architecture parameters (layers, d_model, heads, FFN size, dropout).
        Returns a config dictionary for building TimeSeriesTransformer.
        """
        # ===== Step 1: sample base architecture parameters =====
        n_layers = IntParam(1, 3).sample(trial, "n_layers")
        d_model = IntParam(256, 512, step=64).sample(trial, "d_model")
        n_heads = CategoricalParam([2, 4]).sample(trial, "n_heads")

        # ===== Step 2: sample FFN dimension =====
        m_factor = CategoricalParam([2.0, 2.5, 3.0]).sample(trial, "ffn_multiplier")
        d_ff = int(m_factor * d_model)

        # ===== Step 3: sample dropout parameters =====
        dropout_enabled = CategoricalParam(["On", "Off"]).sample(trial, "dropout")
        dropout = {"enabled": dropout_enabled == "On"}
        if dropout["enabled"]:
            dropout["rate"] = FloatParam(0.1, 0.5, step=0.1).sample(
                trial, "dropout_rate"
            )
        else:
            dropout["rate"] = 0.0

        # ===== Return structured config =====
        return {
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }

    def build_model(
        self, hp: dict[str, Any], input_shape: tuple[int, ...], output_dim: int
    ):
        """
        Instantiate TimeSeriesTransformer_v2 with the given config and perform forward validation.
        Input: input_shape=(C,L); output dimension is output_dim.
        """
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must be (C, L), but got:  {input_shape}")

        _, seq_len = input_shape

        # ===== 6. Instantiate the model =====
        model = TimeSeriesTransformer_v2(
            num_classes=output_dim,
            seq_len=seq_len,
            n_layers=hp["n_layers"],
            d_model=hp["d_model"],
            n_heads=hp["n_heads"],
            d_ff=hp["d_ff"],
            dropout=hp["dropout"]["rate"],
        )

        # Validate the model.
        self.validate_model(model, input_shape, output_dim)
        print("Built Transformer model")

        return model


# ============= Complete usage example =============


def objective(trial):
    """Example Optuna objective: sample a config, build the model, and return an evaluation metric (random value in this example)."""
    builder = CNNHyperBuilder()
    input_shape = (3, 16)  # 1D time series (C, L)
    output_dim = 10
    config = builder.sample_hp(trial)
    _ = builder.build_model(config, input_shape, output_dim)

    # Training and evaluation (pseudo-code)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # accuracy = train_and_evaluate(model, optimizer, train_loader, val_loader)

    # Return a simulated accuracy for demonstration purposes.
    import random

    accuracy = random.random()

    return accuracy


if __name__ == "__main__":
    import optuna

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
