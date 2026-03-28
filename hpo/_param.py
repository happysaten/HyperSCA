from optuna.trial import Trial
from abc import ABC, abstractmethod
from typing import Sequence, Any

# ============= Base parameter classes =============
class _AbstractParam(ABC):
    """Abstract base class for parameter spaces."""

    @abstractmethod
    def sample(self, trial: Trial, name: str) -> Any:
        """Sample a parameter value."""
        raise NotImplementedError


class FixedParam(_AbstractParam):
    """Fixed parameter."""

    def __init__(self, value: Any):
        self.value = value

    def sample(self, trial: Trial, name: str) -> Any:
        return self.value


class CategoricalParam(_AbstractParam):
    """Categorical parameter."""

    def __init__(self, choices: Sequence[Any]):
        self.choices = choices

    def sample(self, trial: Trial, name: str) -> Any:
        return trial.suggest_categorical(name, self.choices)


class IntParam(_AbstractParam):
    """Integer parameter."""

    def __init__(self, low: int, high: int, *, step: int = 1, log: bool = False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def sample(self, trial: Trial, name: str) -> int:
        return trial.suggest_int(
            name, self.low, self.high, step=self.step, log=self.log
        )


class FloatParam(_AbstractParam):
    """Floating-point parameter."""

    def __init__(
        self, low: float, high: float, *, step: float | None = None, log: bool = False
    ):
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def sample(self, trial: Trial, name: str) -> float:
        return trial.suggest_float(
            name, self.low, self.high, step=self.step, log=self.log
        )
