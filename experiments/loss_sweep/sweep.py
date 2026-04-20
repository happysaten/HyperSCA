"""
Run a multi-parameter Cartesian sweep and execute training.
- Repeat training for each parameter set, save each output, and record success/failure.
Usage: call run_sweep(...) or run as a script (see the bottom example).
"""

import itertools
from pathlib import Path
from datetime import datetime

import train


# Common learning-rate candidates (string form for direct argument passing).
DEFAULT_LR_VALUES = ["3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]


def run_sweep(
    param_grid: dict[str, list] | None = None,
    repeat: int = 5,
    output_dir: Path = Path("outputs/sweep"),
    config_file: Path | None = None,
    **extra_kwargs,
):
    # Use the default lr candidate set if param_grid is not provided.
    param_grid = param_grid or {"lr": DEFAULT_LR_VALUES}
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Generate the Cartesian product of parameters.
    combinations = list(itertools.product(*param_values)) if param_values else [tuple()]
    # Ensure the output directory exists and generate a timestamp for the summary file name.
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    total_runs = len(combinations) * repeat
    run_idx = 0

    # Iterate over each parameter combination and repeat training.
    for combo_idx, combo_values in enumerate(combinations, 1):
        combo = dict(zip(param_names, combo_values))
        combo_parts = [f"{k}={combo[k]}" for k in param_names]
        combo_name = "/".join(combo_parts) or "no_params"

        value_dir = output_dir
        for part in combo_parts:
            value_dir /= part

        for r in range(repeat):
            run_idx += 1
            print(f"\n{'=' * 60}")
            print(f"[{run_idx}/{total_runs}] {combo_name}, run {r + 1}/{repeat}")
            print(f"{'=' * 60}\n")

            # Create a unique output directory for this run.
            run_output_dir = value_dir / f"run_{r + 1}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            # Build and inject training arguments (including the parameter combination and extra kwargs).
            train_kwargs = {
                **extra_kwargs,
                "output_dir": run_output_dir,
            }
            train_kwargs.update(combo)

            try:
                train.run(config_file, **train_kwargs)
                return_code = 0
            except Exception as exc:
                # Record the training failure and continue with other runs.
                print(f"Train failed: {exc}")
                return_code = 1

            results.append(
                {
                    "combo": combo,
                    "run": r + 1,
                    "output_dir": run_output_dir,
                    "return_code": return_code,
                }
            )

    # Print and save success statistics for each combination.
    print(f"\n{'=' * 60}")
    print("Sweep Summary")
    print(f"{'=' * 60}")
    for combo_values in combinations:
        combo = dict(zip(param_names, combo_values))
        combo_name = "/".join(f"{k}={v}" for k, v in combo.items()) or "no_params"
        combo_results = [r for r in results if r["combo"] == combo]
        success = sum(1 for r in combo_results if r["return_code"] == 0)
        print(f"{combo_name}: {success}/{repeat} succeeded")

    # Save the summary.
    summary_path = output_dir / f"sweep_summary_{timestamp}.txt"
    with summary_path.open("w") as f:
        f.write(f"Sweep grid: {param_grid}\n")
        f.write(f"Repeat: {repeat}\n\n")
        for r in results:
            status = "OK" if r["return_code"] == 0 else "FAIL"
            combo_desc = (
                "/".join(f"{k}={v}" for k, v in r["combo"].items()) or "no_params"
            )
            f.write(f"{combo_desc} run_{r['run']}: {status}\n")
            f.write(f"  Output: {r['output_dir']}\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    # Example: demo parameter configuration when running the script directly.
    # optimizer_list = ["AdamW", "SGD", "RMSprop"]
    lr_list = [3e-4, 1e-3, 3e-3, 1e-2]
    optimizer_list = ["SGD"]
    param_grid = {"optimizer": optimizer_list, "lr": lr_list}
    run_sweep(
        param_grid=param_grid,
        repeat=5,
        loss_type="HW_BD",
        val_size=0.3,
        batch_size=32,
        epochs=50,
        log_interval=16,
    )
