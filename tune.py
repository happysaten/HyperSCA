"""Optuna hyperparameter search entry point.

Responsibilities include sampling the search space, training candidate models, running attack evaluation, and recording experiment state.
"""

import optuna
import optunahub
from torch.cuda import OutOfMemoryError
import copy
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias
import logging

import train
from eval import test
from hpo.builder import (
    MLPHyperBuilder,
    CNNHyperBuilder,
    TransformerHyperBuilder,
    TrainerHyperBuilder,
    _BaseHyperBuilder,
)
from configs.config import Config, get_config
from dataloaders.load import TorchDatasetBundle

logger = logging.getLogger(__name__)

SamplerType: TypeAlias = Literal["Auto", "Random", "Grid", "TPE", "TPE_MG", "CMA", "GP"]


def get_sampler(
    sampler_type: SamplerType = "TPE",
) -> optuna.samplers.BaseSampler:
    """Build an Optuna sampler by name."""
    match sampler_type:
        case "Auto":
            pkg = optunahub.load_module(package="samplers/auto_sampler")
            assert isinstance(pkg, optuna.samplers.BaseSampler), (
                "Auto sampler must be a subclass of BaseSampler."
            )
            return pkg
        case "Random":
            return optuna.samplers.RandomSampler()
        case "Grid":
            # return optuna.samplers.GridSampler()
            raise NotImplementedError("Grid sampler is not supported.")
        case "TPE":
            return optuna.samplers.TPESampler()
        case "TPE_MG":
            return optuna.samplers.TPESampler(
                multivariate=True, group=True, warn_independent_sampling=False
            )
        case "CMA":
            # return optuna.samplers.CmaEsSampler()
            raise NotImplementedError("CMA-ES sampler is not supported.")
        case "GP":
            return optuna.samplers.GPSampler(deterministic_objective=True)
        case _:
            raise ValueError(f"Invalid sampler type: {sampler_type}")


PrunerType: TypeAlias = Literal["None", "Median", "SuccessiveHalving", "Hyperband"]


def get_pruner(
    pruner_type: PrunerType = "Median",
) -> optuna.pruners.BasePruner:
    """Build an Optuna pruner by name."""
    match pruner_type:
        case "None":
            return optuna.pruners.NopPruner()
        case "Median":
            return optuna.pruners.MedianPruner()
        case "SuccessiveHalving":
            return optuna.pruners.SuccessiveHalvingPruner()
        case "Hyperband":
            return optuna.pruners.HyperbandPruner()
        case _:
            raise ValueError(f"Invalid pruner type: {pruner_type}")


class Objective:
    def __init__(
        self,
        model_builder: _BaseHyperBuilder,
        trainer_builder: TrainerHyperBuilder,
        cfg: Config,
        data: TorchDatasetBundle,
    ) -> None:
        self.model_builder = model_builder
        self.trainer_builder = trainer_builder
        self.cfg = cfg
        self.data = data
        self.start_time = datetime.now()

    def __call__(self, trial: optuna.Trial) -> float:
        """Full flow for a single trial: sample, train, validate, and attack-evaluate."""
        # model = get_model(cfg.model.model_name)(num_classes=cfg.model.num_classes)
        # builder = CNNHyperBuilder()
        cfg = copy.deepcopy(self.cfg)

        # 1) Sample model-structure hyperparameters and instantiate the model for the input shape.
        model_hp = self.model_builder.sample_hp(trial)
        input_shape = self.data["train_ds"][0][0].shape
        model = self.model_builder.build_model(model_hp, input_shape, cfg.num_classes)

        # 2) Sample training hyperparameters and merge them into the current config copy.
        trainer_hp = self.trainer_builder.sample_hp(trial)

        # 3) Train first to obtain the best validation score and matching weights.
        best_sr, best_model = train.train(
            model=model, data=self.data, cfg=cfg.override(**trainer_hp)
        )
        logger.info(f"Trial {trial.number} Best Val SR: {best_sr:.2f}")
        if best_model is None:
            return best_sr

        # 4) If the training stage saved the best weights, load them and run attack evaluation.
        model.load_state_dict(best_model)
        metrics = test(model=model, data=self.data, cfg=cfg.override(**trainer_hp))

        # 5) GE/NTGE are used to measure attack performance and are written into the trial metadata.
        ge = metrics["median_rank"].iloc[-3:].mean().item()
        trial.set_user_attr("ge", ge)
        if ge > cfg.attack_ge_threshold:
            metrics.to_csv(
                f"{cfg.output_dir}/failed_attack_metrics_trial_{trial.number}.csv",
                index=False,
            )
            trial.set_user_attr("success", False)
            logger.info(
                f"Trial {trial.number} Attack failed with GE: {ge:.2f} >= threshold."
            )
        else:
            metrics.to_csv(
                f"{cfg.output_dir}/success_attack_metrics_trial_{trial.number}.csv",
                index=False,
            )
            trial.set_user_attr("success", True)
            if trial.study.user_attrs.get("success") is not True:
                trial.study.set_user_attr("success", True)
                convergence_time = (datetime.now() - self.start_time).total_seconds()
                trial.study.set_user_attr("convergence_time", convergence_time)
            logger.info(
                f"Trial {trial.number} Attack success with GE: {ge:.2f} <= threshold."
            )
            logger.info(f"Model architecture: {model}")

            # Find the first position where median_rank <= 1 and read the corresponding num_traces.
            mask = metrics["median_rank"] <= 1
            ntge = (
                metrics["num_traces"].iloc[mask.idxmax()].item()
                if mask.any()
                else float("nan")
            )
            trial.set_user_attr("ntge", ntge)
            logger.info(f"Trial {trial.number} NTGE: {ntge:.2f}")
            if ntge <= cfg.attack_ntge_threshold:
                logger.info(
                    f"Stopping study as ntge: {ntge:.2f} <= threshold ({cfg.attack_ntge_threshold})."
                )
                trial.study.stop()

        return best_sr


def run(
    model_builder: _BaseHyperBuilder,
    trainer_builder: TrainerHyperBuilder,
    sampler: SamplerType = "TPE",
    pruner: PrunerType = "None",
    timeout: int = 1800,
    config_file: str | None = None,
    output_dir: str = "outputs/optuna",
    **extra_kwargs,
) -> None:
    """Create a study, prepare data, and run a full Optuna search."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Persist study-level logs for later inspection of each trial.
    logging.basicConfig(
        filename=output_dir + "/study.log",
        level=logging.INFO,
        filemode="a",
        format="[%(levelname)s]-%(asctime)s-%(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )

    # Route everything through the root logger to avoid Optuna's default stderr output.
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    cfg = get_config(config_file, output_dir=output_dir, **extra_kwargs)
    logger.info(f"Config:\n{cfg.to_toml()}")

    # Build the study: sampler, pruner, storage, and optimization direction are fixed here.
    study = optuna.create_study(
        sampler=get_sampler(sampler),
        pruner=get_pruner(pruner),
        direction="maximize",
        storage=f"sqlite:///{output_dir}/study.db",
        study_name="study",
    )

    # The objective encapsulates sampling -> training -> restoring best weights -> attack evaluation.
    objective = Objective(
        model_builder=model_builder,
        trainer_builder=trainer_builder,
        cfg=cfg,
        data=train.gen_torch_dataset(cfg),
    )
    study.optimize(
        objective,
        timeout=timeout,
        catch=(OutOfMemoryError,),
        gc_after_trial=True,
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    # ===== Experiment configuration =====
    loss_type = "HW_BD"
    sampler, pruner = "TPE", "None"
    dataset = "ASCAD_R"
    # The default builder is Transformer; keep the line below as a replacement entry point.
    model_builder = TransformerHyperBuilder()
    trainer_builder = TrainerHyperBuilder()
    num_runs = 1
    timeout = 3600  # Maximum runtime per run, in seconds.
    # =====================

    # Name output directories by model + dataset + loss + sampler + pruner + timestamp.
    root_dir = (
        f"outputs/{model_builder.__class__.__name__}_{dataset}_{loss_type}_{sampler}_{pruner}/"
        + datetime.now().strftime("_%m%d_%H%M%S")
    )
    base_path = Path(root_dir)
    dataset_config_file = f"configs/{dataset}.toml"

    # Global task markers used to determine whether the batch is still running.
    total_running_marker = base_path / "TOTAL_RUNNING.marker"
    total_finished_marker = base_path / "TOTAL_FINISHED.marker"

    def get_now_str() -> str:
        """Return a consistently formatted local time string for marker files."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the output root directory first, then write the global running marker.
    base_path.mkdir(parents=True, exist_ok=True)
    # Use overwrite on the first write to ensure a clean initial state.
    total_running_marker.write_text(f"Total Task Started at: {get_now_str()}\n")

    try:
        for i in range(num_runs):
            run_dir = base_path / f"run_{i + 1}"
            running_marker = run_dir / "RUNNING.marker"
            finished_marker = run_dir / "FINISHED.marker"
            error_marker = run_dir / "ERROR.log"

            # Each run gets its own directory to isolate logs, errors, and state files.
            run_dir.mkdir(parents=True, exist_ok=True)

            # Record the subtask start time to help diagnose hangs or early termination.
            start_time = get_now_str()
            # Append entries to preserve the full batch history.
            with total_running_marker.open("a") as f:
                f.write(f"Run {i + 1}/{num_runs} Started at: {start_time}\n")
            with running_marker.open("a") as f:
                f.write(f"Started at: {start_time}\n")

            try:
                # Single execution: pass the current run configuration to the Optuna search entry point.
                run(
                    model_builder=model_builder,
                    trainer_builder=trainer_builder,
                    sampler=sampler,
                    pruner=pruner,
                    timeout=timeout,
                    config_file=dataset_config_file,
                    output_dir=str(run_dir),
                    loss_type=loss_type,
                )
            except Exception as _:
                # Capture exceptions and save the stack trace so one failure does not lose context.
                import traceback

                # Append so the error file preserves the full history.
                with error_marker.open("a") as f:
                    f.write(
                        f"Error occurred in run {i + 1} at {get_now_str()}:\n{traceback.format_exc()}\n"
                    )
                logger.error(f"Run {i + 1} failed! Check {error_marker} for details.")
            finally:
                from torch.cuda import empty_cache  # ⭐ removed

                # Clear CUDA cache to avoid fragmented memory carrying over to the next run.
                empty_cache()

                # Subtask finished: write the end time and then rename the running marker.
                if running_marker.exists():
                    # Record the end time before renaming the marker.
                    with running_marker.open("a") as f:
                        f.write(f"Ended at: {get_now_str()}\n")
                    running_marker.replace(finished_marker)

    finally:
        # Regardless of what happens, close the global running marker at the end.
        if total_running_marker.exists():
            content = total_running_marker.read_text()
            # Append the global end time before switching to the finished marker.
            with total_running_marker.open("a") as f:
                f.write(f"Total Task Ended at: {get_now_str()}\n")
            total_running_marker.replace(total_finished_marker)
