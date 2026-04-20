"""Attack evaluation: compute key ranks and aggregate key metrics (mean/median/top1/top3)."""

import numpy as np
import pandas as pd
from collections.abc import Iterator, Iterable


def calc_key_metrics(
    scores: np.ndarray,
    median: np.ndarray,
    key: int,
    num_traces_list: Iterable[int],
    repeat: int = 100,
) -> pd.DataFrame:
    """Compute key-rank statistics for different sample counts.

    Args:
        scores: Score matrix, shape (N, K).
        median: Reordering index for each trace (length N).
        key: Index of the true key.
        num_traces_list: List of trace counts to evaluate.
        repeat: Resampling count per trace count (default 100).

    Returns:
        pd.DataFrame: Aggregated metrics for each num_traces value (num_traces, mean_rank, median_rank, top1, top3).
    """
    assert len(median) == len(scores), (
        f"Length mismatch: {len(median)} vs {len(scores)}"
    )
    # scores_realignd = scores[np.arange(shape[0])[:, None], median]
    scores_realignd = np.ascontiguousarray(np.take_along_axis(scores, median, axis=1))
    # scores_realignd = realign_scores(scores, meta, is_hw, method="vectorized")
    ret = []
    for num_traces in (
        # tqdm(num_traces_list, desc="Calculating key metrics")
        # if len(num_traces_list) > 10
        # else num_traces_list
        num_traces_list
    ):
        ranks = ranking(scores_realignd, key, num_traces, repeat)
        ret.append(ret_ranking(ranks, num_traces))
    return pd.DataFrame(ret)


# TODO:
# 1. Allow njit for performance gains, but verify the benefit with benchmarking first.
# 2. Consider the parallel option and prange.
# 3. Consider using pseudo-random indices.
# 4. Consider how NaN handling and different aggregation modes such as mean vs sum affect the result, and axis=0 vs axis=1.
# 5. Use a profiler to analyze performance bottlenecks.


# @njit(parallel=True)
def ranking(
    key_scores: np.ndarray,
    key_true: int,
    num_traces: int,
    repeat: int,
) -> np.ndarray:
    """Compute the key-rank distribution across repeated trials using sampling without replacement.

    Args:
        key_scores: Reordered score matrix, shape (M, K).
        key_true: Index of the true key.
        num_traces: Number of samples per draw.
        repeat: Number of repeated trials.

    Returns:
        np.ndarray: Rank array of length repeat (may be float).
    """
    idx_gen = idx_generator(len(key_scores), num_traces, repeat)
    ranks = []
    for idx in idx_gen:
        scores = np.nanmean(key_scores[idx], axis=0)
        # scores = np.mean(key_scores[idx], axis=0)
        # scores = np.sum(key_scores[idx], axis=0)
        # scores = np.nansum(key_scores[idx], axis=0)
        socres_cmp = scores - scores[key_true]

        # Compute rank (smaller is better).
        # rank = (socres_cmp > 0).sum() + ((socres_cmp == 0).sum() - 1) / 2
        rank = (np.sign(socres_cmp).sum() + len(scores) - 1) / 2
        ranks.append(rank)

    # ranks = np.array(ranks)
    # return ret_ranking(ranks, num_traces)
    return np.array(ranks)


def ret_ranking(ranks: np.ndarray, num_traces: int) -> dict:
    """Aggregate metrics from the repeated-trial rank array and return a dictionary."""
    return {
        "num_traces": num_traces,
        "mean_rank": ranks.mean(),
        "median_rank": np.median(ranks),
        "top1": np.mean(ranks == 0),
        "top3": np.mean(ranks <= 2),
    }


def idx_generator(M: int, n: int, repeat: int) -> Iterator[np.ndarray]:
    """
    Independent random-sampling generator without replacement.
    - Pre-generate all permutations.
    - No explicit seed; use the system entropy source.
    """
    rng = np.random.default_rng()
    idx_matrix = np.array([rng.permutation(M)[:n] for _ in range(repeat)])

    for i in range(repeat):
        yield idx_matrix[i]
