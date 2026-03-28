"""Attack evaluation backed by torch: compute key ranks and aggregate key metrics (mean/median/top1/top3)."""

import torch
import pandas as pd
from tqdm import tqdm


def calc_key_metrics(
    scores: torch.Tensor,
    median: torch.Tensor,
    key: int,
    num_traces_list: list[int],
    repeat: int = 100,
    # metrics: Literal["mean_rank", "median_rank", "top1", "top3"] = "top1",
) -> pd.DataFrame:
    assert len(median) == len(scores), f"Length mismatch: {len(median)} vs {len(scores)}"
    # scores_realignd = scores[np.arange(shape[0])[:, None], median]
    # scores_realignd = np.ascontiguousarray(np.take_along_axis(scores, median, axis=1))
    scores_realignd = torch.take_along_dim(scores, median, dim=1).contiguous()
    # scores_realignd = realign_scores(scores, meta, is_hw, method="vectorized")
    ret = []
    for num_traces in (
        tqdm(num_traces_list, desc="Calculating key metrics")
        if len(num_traces_list) > 10
        else num_traces_list
    ):
        ranks = ranking(scores_realignd, key, num_traces, repeat)
        ret.append(ret_ranking(ranks, num_traces))
    return pd.DataFrame(ret)

@torch.compile
def ranking(
    key_scores: torch.Tensor,
    key_true: int,
    num_traces: int,
    repeat: int,
) -> torch.Tensor:
    idx_gen = idx_generator(len(key_scores), num_traces, repeat)
    ranks = []
    for idx in idx_gen:
        scores = torch.nanmean(key_scores[idx], dim=0)
        socres_cmp = scores - scores[key_true]

        # Compute rank (smaller is better).
        # rank = (socres_cmp > 0).sum() + ((socres_cmp == 0).sum() - 1) / 2
        rank = (torch.sign(socres_cmp).sum() + len(scores) - 1) / 2
        ranks.append(rank)

    return torch.tensor(ranks)

def ret_ranking(ranks: torch.Tensor, num_traces: int) -> dict:
    return {
        "num_traces": num_traces,
        "mean_rank": ranks.mean(),
        "median_rank": torch.median(ranks),
        "top1": torch.mean((ranks == 0).to(dtype=torch.float)),
        "top3": torch.mean((ranks <= 2).to(dtype=torch.float)),
    }


def idx_generator(M: int, n: int, repeat: int):
    """
    Independent random-sampling generator without replacement.
    - Pre-generate all permutations.
    - No explicit seed; use the system entropy source.
    """
    idx_matrix = torch.stack([torch.randperm(M)[:n] for _ in range(repeat)])

    for i in range(repeat):
        yield idx_matrix[i]

