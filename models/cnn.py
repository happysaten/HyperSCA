"""Lightweight and deep 1D-CNN model definitions.
Convention: input tensors have shape (N, C, L), and outputs are logits (N, num_classes).
Usage: SimpleCNN is suited to quick validation and small models; DeepCNN is suited to more complex feature extraction scenarios.
"""

from torch import nn
import torch


class SimpleCNN(nn.Module):
    """Simple 1D convolutional network: Conv->ELU->Pool->FC sequence."""

    def __init__(self, num_classes: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=26, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(576),
            nn.ELU(inplace=True),
            nn.Linear(576, 960),
            nn.ELU(inplace=True),
            nn.Linear(960, num_classes),
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x (N, C, L) -> logits (N, num_classes). Accelerated by torch.compile."""
        return self.net(x)


class DeepCNN(nn.Module):
    """Deeper 1D convolutional network: multiple Conv+Pool blocks plus fully connected layers and Dropout for more complex tasks."""

    def __init__(self, num_classes: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 960),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(960, num_classes),
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x (N, C, L) -> logits (N, num_classes). Accelerated by torch.compile."""
        return self.net(x)
