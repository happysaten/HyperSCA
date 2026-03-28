import h5py
from pathlib import Path
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import TypeAlias, TypedDict

from utils.aes import get_sout, get_hw, BYTES, multGF2_8, get_sin
from configs.config import Config

rng = random.Random()

DatasetSplit: TypeAlias = tuple[np.ndarray, np.ndarray, int, np.ndarray]
DatasetBundle: TypeAlias = tuple[DatasetSplit, DatasetSplit]


class TorchDatasetBundle(TypedDict):
    """Complete return value of gen_torch_dataset()."""

    train_ds: TensorDataset
    val_ds: TensorDataset
    test_ds: TensorDataset
    key_val: int
    median_val: np.ndarray
    key_test: int
    median_test: np.ndarray


def gen_torch_dataset(
    cfg: Config,
) -> TorchDatasetBundle:
    """
    Load and return the training/validation/test sets, along with the key and median needed for attack evaluation.

    Returned fields:
    - train_ds / val_ds / test_ds: TensorDataset
    - key_val / key_test: corresponding keys
    - median_val / median_test: corresponding median labels
    """
    # Read the raw profiling and attack data, then perform the basic reconstruction.
    (
        (prf_traces, prf_labels, prf_key, prf_median),
        (atk_traces, atk_labels, atk_key, atk_median),
    ) = load_dataset(
        dataset_type=cfg.dataset_type,
        file_path=cfg.dataset_path,
        prf_num=cfg.prf_num,
        atk_num=cfg.atk_num,
        target_byte=cfg.target_byte,
        standardize=cfg.standardize,
        random_select=cfg.random_select,
        is_hw=cfg.is_hw,
    )

    # The validation source is controlled by the config: profiling split or attack directly.
    if cfg.val_source == "profiling":
        key_val = int(prf_key)
        X_train, X_val, y_train, y_val, _, median_val = train_test_split(
            prf_traces,
            prf_labels,
            prf_median,
            test_size=cfg.val_size,
            random_state=cfg.random_state,
            stratify=prf_labels,
        )
    else:
        key_val = int(atk_key)
        X_train, y_train = prf_traces, prf_labels
        X_val, y_val, median_val = atk_traces, atk_labels, atk_median

    # The test set always uses the attack split to keep attack evaluation consistent.
    X_test, y_test = atk_traces, atk_labels
    key_test, median_test = int(atk_key), atk_median

    # Normalize the input shape to (N, C, L).
    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]  # (N, L) -> (N, 1, L)
        X_val = X_val[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

    # If the shape is (N, L, C), transpose to (N, C, L); keep it unchanged if it is already (N, C, L).
    elif X_train.ndim == 3:
        if X_train.shape[2] < X_train.shape[1]:
            X_train = np.transpose(X_train, (0, 2, 1))
            X_val = np.transpose(X_val, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))

    else:
        raise ValueError(
            f"Unsupported X shape: {X_train.shape}. Expected a 2D or 3D array."
        )

    # Normalize labels to 1D.
    if y_train.ndim != 1:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
    if y_test.ndim != 1:
        y_test = y_test.ravel()

    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    # Convert to PyTorch tensors and build TensorDataset instances.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.uint8)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.uint8)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.uint8)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "key_val": key_val,
        "median_val": median_val,
        "key_test": key_test,
        "median_test": median_test,
    }


def load_dataset(dataset_type: str, **kwargs) -> DatasetBundle:
    """Dispatch to the appropriate loader by dataset type."""
    match dataset_type:
        case "ASCAD_F" | "ASCAD_R" | "CHES_CTF" | "AES_RD":
            return load_sca(**kwargs)
        case "ASCAD_v2":
            return load_ascad_v2(**kwargs)
        case "AES_HD":
            return load_aes_hd(**kwargs)
        case _:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_sca(
    file_path: Path,
    prf_num: int = -1,
    atk_num: int = -1,
    target_byte: int = 2,
    standardize: bool = True,
    random_select: bool = True,
    is_hw: bool = False,
) -> DatasetBundle:
    """Load a generic SCA dataset and return profiling and attack splits."""
    with h5py.File(file_path, "r") as f:
        prf_traces = f["Profiling_traces/traces"][:]
        prf_meta = f["Profiling_traces/metadata"]
        prf_plain = prf_meta["plaintext"][:, target_byte].astype(np.uint8)
        prf_key = prf_meta["key"][:, target_byte].astype(np.uint8)

        # Select profiling sample indices as needed.
        prf_index = choice_index(len(prf_traces), prf_num, random_select)

        # Filter samples and map non-constant keys to one random key.
        prf_traces, prf_plain, prf_key = (
            prf_traces[prf_index],
            prf_plain[prf_index],
            prf_key[prf_index],
        )
        assert prf_plain.ndim == 1
        if np.any(prf_key != prf_key[0]):
            new_key = rng.randint(0, 255)
            prf_plain ^= prf_key ^ new_key
            prf_key = new_key
        else:
            prf_key = int(prf_key[0])

        # Compute labels and medians; switch to Hamming Weight when needed.
        prf_labels = get_sout(prf_plain ^ prf_key)
        prf_median = get_sout(prf_plain[:, None] ^ BYTES)
        if is_hw:
            prf_labels = get_hw(prf_labels)
            prf_median = get_hw(prf_median)

        # Fit the scaler on profiling data and then apply it to attack data.
        if standardize:
            scaler = StandardScaler()
            prf_traces = scaler.fit_transform(prf_traces)

        # Reuse the same processing flow for the attack split.
        atk_traces = f["Attack_traces/traces"][:]
        atk_meta = f["Attack_traces/metadata"]
        atk_plain = atk_meta["plaintext"][:, target_byte].astype(np.uint8)
        atk_key = atk_meta["key"][:, target_byte].astype(np.uint8)
        atk_index = choice_index(len(atk_traces), atk_num, random_select)
        atk_traces, atk_plain, atk_key = (
            atk_traces[atk_index],
            atk_plain[atk_index],
            atk_key[atk_index],
        )
        assert atk_plain.ndim == 1
        if np.any(atk_key != atk_key[0]):
            new_key = rng.randint(0, 255)
            atk_plain ^= atk_key ^ new_key
            atk_key = new_key
        else:
            atk_key = int(atk_key[0])
        atk_labels = get_sout(atk_plain ^ atk_key)
        atk_median = get_sout(atk_plain[:, None] ^ BYTES)
        assert np.all(atk_median[:, atk_key] == atk_labels)
        if is_hw:
            atk_labels = get_hw(atk_labels)
            atk_median = get_hw(atk_median)
        if standardize:
            atk_traces = scaler.transform(atk_traces)

    print(
        f"Profiling traces shape: {prf_traces.shape}, Attack traces shape: {atk_traces.shape}"
    )

    return (prf_traces, prf_labels, prf_key, prf_median), (
        atk_traces,
        atk_labels,
        atk_key,
        atk_median,
    )


def load_ascad_v2(
    file_path: Path,
    prf_num: int = -1,
    atk_num: int = -1,
    target_byte: int = 0,
    standardize: bool = True,
    random_select: bool = True,
    is_hw: bool = False,
) -> DatasetBundle:
    """Load the ASCAD v2 dataset and reconstruct inputs and intermediates from multi-label data."""
    poi = np.arange(5000 + 450, 5000 + 650)
    with h5py.File(file_path, "r") as f:
        prf_traces = f["Profiling_traces/traces"][:]
        prf_labels_multi = f["Profiling_traces/labels"][:]
        prf_meta = f["Profiling_traces/metadata"][:]
        prf_index = choice_index(
            len(prf_traces),
            prf_num,
            random_select,
            index_exclude=[99999, 199999, 299999, 399999, 499999],
        )

        # Reconstruct plaintext/key from metadata using perm_index and crop the POI.
        prf_traces, prf_perm_index = (
            prf_traces[prf_index],
            prf_labels_multi["perm_index"][prf_index],
        )
        prf_plain = prf_meta["plaintext"][prf_index[:, None], prf_perm_index][
            :, target_byte
        ]
        prf_key = prf_meta["key"][prf_index[:, None], prf_perm_index][:, target_byte]
        prf_traces = prf_traces[:, poi]
        assert prf_plain.ndim == 1

        # If the key is not constant, unify it to a single random key and handle the alpha/beta masks.
        if np.any(prf_key != prf_key[0]):
            new_key = rng.randint(0, 255)
            prf_plain ^= prf_key ^ new_key
            prf_key = new_key
        else:
            prf_key = int(prf_key[0])
        prf_alpha_mask = prf_labels_multi["alpha_mask"][prf_index].reshape(-1, 1)
        prf_beta_mask = prf_labels_multi["beta_mask"][prf_index].reshape(-1, 1)
        prf_labels = prf_labels_multi["sbox_masked"][prf_index, target_byte]
        # prf_labels = prf_alpha_mask * get_sout(prf_plain ^ prf_key) ^ prf_beta_mask
        prf_median = (
            multGF2_8(prf_alpha_mask, get_sout(prf_plain[:, None] ^ BYTES))
            ^ prf_beta_mask
        )
        if is_hw:
            prf_labels = get_hw(prf_labels)
            prf_median = get_hw(prf_median)

        # Standardize with the profiling scaler.
        if standardize:
            scaler = StandardScaler()
            prf_traces = scaler.fit_transform(prf_traces)

        # Use the same reconstruction flow for the attack split.
        atk_traces = f["Attack_traces/traces"][:]
        atk_labels_multi = f["Attack_traces/labels"][:]
        atk_meta = f["Attack_traces/metadata"][:]
        atk_index = choice_index(len(atk_traces), atk_num, random_select)

        atk_traces, atk_perm_index = (
            atk_traces[atk_index],
            atk_labels_multi["perm_index"][atk_index],
        )
        atk_plain = atk_meta["plaintext"][atk_index[:, None], atk_perm_index][
            :, target_byte
        ]
        atk_key = atk_meta["key"][atk_index[:, None], atk_perm_index][:, target_byte]
        atk_traces = atk_traces[:, poi]
        assert atk_plain.ndim == 1
        if np.any(atk_key != atk_key[0]):
            new_key = rng.randint(0, 255)
            atk_plain ^= atk_key ^ new_key
            atk_key = new_key
        else:
            atk_key = int(atk_key[0])
        atk_alpha_mask = atk_labels_multi["alpha_mask"][atk_index].reshape(-1, 1)
        atk_beta_mask = atk_labels_multi["beta_mask"][atk_index].reshape(-1, 1)
        atk_labels = atk_labels_multi["sbox_masked"][atk_index, target_byte]
        atk_median = (
            multGF2_8(atk_alpha_mask, get_sout(atk_plain[:, None] ^ BYTES))
            ^ atk_beta_mask
        )
        assert np.all(atk_median[:, atk_key] == atk_labels)
        if is_hw:
            atk_labels = get_hw(atk_labels)
            atk_median = get_hw(atk_median)

        if standardize:
            atk_traces = scaler.transform(atk_traces)

    print(
        f"Profiling traces shape: {prf_traces.shape}, Attack traces shape: {atk_traces.shape}"
    )

    return (prf_traces, prf_labels, prf_key, prf_median), (
        atk_traces,
        atk_labels,
        atk_key,
        atk_median,
    )


def load_aes_hd(
    file_path: Path,
    prf_num: int = -1,
    atk_num: int = -1,
    target_byte: int = 7,
    standardize: bool = True,
    random_select: bool = True,
    is_hw: bool = False,
) -> DatasetBundle:
    """Load the AES_HD dataset and reconstruct labels from round_key/ciphertext."""
    assert target_byte == 7, "AES_HD only supports target_byte=7"
    with h5py.File(file_path, "r") as f:
        prf_traces = f["Profiling_traces/traces"][:]
        prf_meta = f["Profiling_traces/metadata"]
        prf_key = prf_meta["round_key"][:, 7].astype(np.uint8)
        prf_cipher_3, prf_cipher_7 = (
            prf_meta["ciphertext"][:, 3].astype(np.uint8),
            prf_meta["ciphertext"][:, 7].astype(np.uint8),
        )
        prf_index = choice_index(len(prf_traces), prf_num, random_select)

        prf_traces, prf_cipher_3, prf_cipher_7, prf_key = (
            prf_traces[prf_index],
            prf_cipher_3[prf_index],
            prf_cipher_7[prf_index],
            prf_key[prf_index],
        )

        # If the key is not constant, unify it; compute labels and medians (sin XOR cipher_3).
        assert prf_cipher_3.ndim == 1
        if np.any(prf_key != prf_key[0]):
            new_key = rng.randint(0, 255)
            prf_cipher_7 ^= prf_key ^ new_key
            prf_key = new_key
        else:
            prf_key = int(prf_key[0])
        prf_labels = get_sin(prf_cipher_7 ^ prf_key) ^ prf_cipher_3
        prf_median = get_sin(prf_cipher_7[:, None] ^ BYTES) ^ prf_cipher_3[:, None]
        if is_hw:
            prf_labels = get_hw(prf_labels)
            prf_median = get_hw(prf_median)

        if standardize:
            scaler = StandardScaler()
            prf_traces = scaler.fit_transform(prf_traces)

        # Attack: same flow as profiling.
        atk_traces = f["Attack_traces/traces"][:]
        atk_meta = f["Attack_traces/metadata"]
        atk_key = atk_meta["round_key"][:, 7].astype(np.uint8)
        atk_cipher_3, atk_cipher_7 = (
            atk_meta["ciphertext"][:, 3].astype(np.uint8),
            atk_meta["ciphertext"][:, 7].astype(np.uint8),
        )
        atk_index = choice_index(len(atk_traces), atk_num, random_select)
        atk_traces, atk_cipher_3, atk_cipher_7, atk_key = (
            atk_traces[atk_index],
            atk_cipher_3[atk_index],
            atk_cipher_7[atk_index],
            atk_key[atk_index],
        )
        assert atk_cipher_3.ndim == 1
        if np.any(atk_key != atk_key[0]):
            new_key = rng.randint(0, 255)
            atk_cipher_7 ^= atk_key ^ new_key
            atk_key = new_key
        else:
            atk_key = int(atk_key[0])
        atk_labels = get_sin(atk_cipher_7 ^ atk_key) ^ atk_cipher_3
        atk_median = get_sin(atk_cipher_7[:, None] ^ BYTES) ^ atk_cipher_3[:, None]
        assert np.all(atk_median[:, atk_key] == atk_labels)
        if is_hw:
            atk_labels = get_hw(atk_labels)
            atk_median = get_hw(atk_median)
        if standardize:
            atk_traces = scaler.transform(atk_traces)

    print(
        f"Profiling traces shape: {prf_traces.shape}, Attack traces shape: {atk_traces.shape}"
    )

    return (prf_traces, prf_labels, prf_key, prf_median), (
        atk_traces,
        atk_labels,
        atk_key,
        atk_median,
    )


def choice_index(
    M: int, N: int, random_select: bool = True, index_exclude: list[int] | None = None
) -> np.ndarray:
    """
    Select indices from [0, M).

    Rules:
    - Support excluding positions via index_exclude.
    - If N <= 0 or N >= the number of available items, return all available indices.
    - When random_select=True, sample randomly and then sort; otherwise take the first N items.
    """
    # Build the candidate index pool first.
    if index_exclude is not None and len(index_exclude) > 0:
        candidates = np.delete(np.arange(M), index_exclude)
    else:
        candidates = np.arange(M)

    # If N is invalid, return all candidate indices directly.
    if N >= len(candidates) or N <= 0:
        return candidates

    # Sample randomly or sequentially according to the config.
    if random_select:
        selected = np.array(rng.sample(candidates.tolist(), N))
        selected.sort()  # Sort in place to keep the output stable.
        return selected
    else:
        return candidates[:N]
