**HyperSCA** is a hyperparameter optimization framework for deep learning-based side-channel attacks. It targets known-plaintext/profiling-based side-channel analysis research, emphasizing automated hyperparameter tuning and sca-aware loss functions to enhance attack performance.

The corresponding paper for this project:

**HyperSCA: A Comprehensive Hyperparameter Optimization Framework for Deep Learning-based Side Channel Attacks**

---

## 📂 Project Structure

The project adopts a modular design with clear separation of core logic:

```text
HyperSCA/
├── experiments/      # Parameter sweep and experiment visualization tools
├── configs/          # Dataset and experiment configuration files (TOML format)
├── dataloaders/      # Dataset loading and preprocessing helper code
├── models/           # CNN, Transformer, and custom loss implementations
├── hpo/              # Hyperparameter builders for Optuna search
├── utils/            # Attack metrics, AES utilities, logging, and tensor tools
├── train.py          # Training entry point
├── tune.py           # Optuna tuning entry point
├── eval.py           # Evaluation and plotting entry point
└── run_tune.py       # Script for batch launching tuning processes
```

---

## 🚀 Quick Start

### Installation

This repository does not include a pre-created `.venv` directory. Please create the local environment using `uv`:

```bash
uv sync
```

To explicitly execute commands within the project environment:

```bash
uv run python --version
```

Dependencies are defined in [pyproject.toml](pyproject.toml).

### Training a Model

```bash
uv run python train.py
```

By default, `train.py` uses the built-in configuration in its main function.

### Running Hyperparameter Tuning

```bash
uv run python tune.py
```

`tune.py` launches the Optuna-based optimization process, trains candidate models, evaluates attack metrics, and saves experimental artifacts to the configured output directory.

### Evaluating Trained Models

```bash
uv run python eval.py
```

The evaluation entry point loads saved models, performs attack evaluation, and can generate metric plots.

### Batch Launching Tuning Tasks

```bash
uv run python run_tune.py 4 "[0,1]"
```

This helper script can launch multiple tuning processes and assign them to specified GPUs. To stop matching tuning tasks:

```bash
uv run python run_tune.py stop
```

---

## ⚙️ Configuration

Project-level dataset and experiment configurations are located in the [configs/](configs/) directory.

Typical configuration workflow:
1. Select a dataset configuration, e.g., `configs/ASCAD_R.toml`.
2. Adjust model, optimizer, loss function, and batch-related parameters in the configuration file.
3. Run `train.py`, `tune.py` directly, or import relevant functions in custom scripts.

The configuration system centers around [configs/config.py](configs/config.py), which is reused by both training and evaluation entry points.

---

## 📚 Datasets

The following dataset sources are used for HyperSCA's experimental setup.

- [**ASCAD fixed-key**](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)
- [**ASCAD variable-key**](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)
- [**AES_HD**](http://aisylabdatasets.ewi.tudelft.nl/aes_hd.h5)
- [**ASCAD v2**](https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2)
- [**CHES CTF 2018 News**](https://chesctf.riscure.com/2018/news)
- [**AES_RD**](https://github.com/ikizhvatov/randomdelays-traces)

---

### 💾 Experimental Logs

Complete experimental results and logs  are available in the [Releases](https://github.com/happysaten/HyperSCA/releases) page; 

---

## 📝 Citation

If you use HyperSCA in academic work, please cite this project.

```bibtex
% TODO: Add final BibTeX entry here after paper official publication.
@article{TODO_HyperSCA,
title   = {HyperSCA: A Comprehensive Hyperparameter Optimization Framework for Deep Learning-based Side Channel Attacks},
author  = {[TODO: Fill in authors]},
journal = {[TODO: Fill in journal or conference]},
year    = {[TODO: Fill in year]},
note    = {[TODO: Fill in DOI, arXiv, or official link]}
}
```