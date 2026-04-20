"""Collection of loss functions and scoring functions.

Includes training losses, inference scoring, and a few reusable tensor constants.
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch._dynamo.config
from scipy.special import comb
from typing import Callable, Literal

from utils.torch_utils import standardization, standardization_
from utils.torch_utils import generate_dist_space

torch._dynamo.config.cache_size_limit = 16

# Common constant tensors: keep them on CUDA to avoid repeated construction.
BYTES = torch.arange(256, dtype=torch.float16, device="cuda")
HWS = torch.arange(9, dtype=torch.float16, device="cuda")
BITS = HWS / 8

# Log binomial coefficients used by the HW_BD score/loss.
COMBS = torch.tensor(
    [comb(8, i) for i in range(9)], dtype=torch.float16, device="cuda"
).log()


def get_loss_and_scoring(
    name: Literal["ID_CE", "HW_CE", "ID_DC", "HW_MSE", "HW_BD"],
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Return the training loss and matching scoring function for the given loss name."""
    match name:
        case "ID_CE" | "HW_CE":
            return (
                nn.CrossEntropyLoss(),
                _cross_entropy_scoring,
            )
        case "ID_DC":
            dist = generate_dist_space("hd", 256)
            return (
                CorrDistLoss(dist),
                CorrDistScoring(dist),
            )
        case "HW_MSE":
            return (
                nn.MSELoss(),
                _mse_scoring,
            )
        case "HW_BD":
            return (
                HWBDLoss(),
                _hw_bd_scoring,
            )


def _cross_entropy_scoring(input: torch.Tensor) -> torch.Tensor:
    """Convert logits into cross-entropy scores suitable for ranking/comparison."""
    return (
        input.log_softmax(dim=1, dtype=torch.double)
        .clamp_min_(-50.0)
        .nan_to_num_(-50.0)
        .type_as(input)
    )


def _mse_scoring(input: torch.Tensor) -> torch.Tensor:
    """Map regression outputs to an MSE score where larger is better."""
    return input.sub(HWS.type_as(input)).square_().neg_()


def _mse_logits_scoring(input: torch.Tensor) -> torch.Tensor:
    """MSE scoring for logits."""
    return input.sigmoid_().sub(BITS.type_as(input)).square_().neg_()


def _l1_logits_scoring(input: torch.Tensor) -> torch.Tensor:
    """L1 scoring for logits."""
    return input.sigmoid_().sub(BITS.type_as(input)).abs_().neg_()


def _hw_bd_scoring(input: torch.Tensor) -> torch.Tensor:
    """Logit scoring for HW_BD loss."""
    # logit = input.sigmoid().double()
    # return HWS * logit.log() + (8 - HWS) * (1 - logit).log() + COMBS
    hws, combs = HWS.type_as(input), COMBS.type_as(input)
    # logit = F.sigmoid(input)
    # return hws * logit.log() + (8 - hws) * ((1 - logit).log()) + combs
    log_logit = F.logsigmoid(input).clamp_min_(-50.0).nan_to_num_(-50.0)
    return hws * log_logit + (8 - hws) * (log_logit - input) + combs


def _acc_scoring(input: torch.Tensor) -> torch.Tensor:
    """Return a one-hot auxiliary score for top-1 accuracy."""
    return F.one_hot(input.argmax(dim=1), num_classes=input.shape[1])


def _rank_scoring(input: torch.Tensor) -> torch.Tensor:
    """Return the rank index for each class."""
    return input.argsort(dim=1, descending=True).argsort(dim=1)


class CorrDistScoring:
    """Correlation-based scorer over a distance space."""

    def __init__(self, dist: torch.Tensor) -> None:
        self.dist = dist.contiguous().T

    # @torch.compile
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the correlation score between the input and the distance space."""
        return torch.mm(
            standardization_(input.log_softmax(dim=1), dim=1),
            self.dist.type_as(input),
            out=input,
        ).neg_()


class CosineSimilarityScoring:
    """Cosine-similarity-based scorer."""

    def __init__(self, dist: torch.Tensor) -> None:
        self.dist = dist

    @torch.compile
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the cosine similarity between the input and the distance space."""
        return F.cosine_similarity(
            input.log_softmax(dim=1)[:, None, :], self.dist.type_as(input), dim=-1
        )


class ContrastiveLoss(nn.Module):
    """Contrastive loss.

    Reference implementation: Ranking-Loss-SCA.
    """

    alpha: torch.Tensor

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        from math import log

        self._bias = log(2.0)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: [B, C], target: [B] -> [B, 1]
        target_score = input.take_along_dim(target.unsqueeze(1), dim=1)

        # diff: [B, C]
        diff = input - target_score

        # 1. Ensure alpha is a Tensor so it can multiply diff (fixes Bug 2).
        # 2. Keep the log(1 + exp(alpha*x)) form so alpha can amplify gradients (resolves the mathematical conflict).
        # 3. Use the built-in numerical stability of F.softplus (fixes overflow issues).
        loss = F.softplus(diff * self.alpha)

        # Subtract the bias.
        # Here self.bias = log(2), so subtracting log(2)/C cancels the target-class floor.
        return loss.mean() - self._bias / input.shape[1]


class CorrDistLoss(nn.Module):
    """Loss based on correlation distance."""

    def __init__(self, dist: torch.Tensor) -> None:
        super().__init__()
        torch.set_float32_matmul_precision("high")
        # Hack for a triton bug - https://github.com/pytorch/pytorch/issues/124565
        torch.empty(1, device="cuda", requires_grad=True).backward()
        self.dist = dist.contiguous()
        self.scale = len(dist) ** 0.5
        # self.scale = 1.0

    @torch.compile
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the correlation-distance loss."""
        # output = torch.mm(input.softmax(), self.dist)
        # return -F.nll_loss(output, target)
        return self.scale * (
            (
                torch.dot(
                    standardization(input.log_softmax(dim=1), dim=1).view(-1),
                    self.dist.type_as(input)[target].view(-1),
                )
            )
            / input.numel()
            + 1.0
        )


class CosineSimilarityLoss(nn.Module):
    """Loss based on cosine similarity."""

    def __init__(self, dist: torch.Tensor) -> None:
        super().__init__()
        torch._dynamo.config.cache_size_limit = 16
        # Hack for a triton bug - https://github.com/pytorch/pytorch/issues/124565
        torch.empty(1, device="cuda", requires_grad=True).backward()
        self.dist = dist.contiguous()

    @torch.compile(dynamic=True)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the cosine-similarity loss."""
        return 1.0 - F.cosine_similarity(
            input.log_softmax(dim=1), self.dist.type_as(input), dim=-1
        )


class HWBDLoss(nn.Module):
    """Hamming-weight binary cross-entropy loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(input, target / 8.0)
