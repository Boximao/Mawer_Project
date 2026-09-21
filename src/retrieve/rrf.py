"""Reciprocal Rank Fusion (k=60) and min-max normalization."""
from __future__ import annotations


def rrf_scores(
    rank_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    weights = weights or [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights must match rank_lists")
    for lst, weight in zip(rank_lists, weights):
        for rank, chunk_id in enumerate(lst, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {cid: 1.0 for cid in scores}
    return {cid: (s - lo) / (hi - lo) for cid, s in scores.items()}


def fuse(
    rank_lists: list[list[str]],
    *,
    k: int = 60,
    cap: int = 30,
    discard_below: float = 0.40,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    raw = rrf_scores(rank_lists, k=k, weights=weights)
    ranked = sorted(raw.items(), key=lambda kv: -kv[1])[:cap]
    capped = dict(ranked)
    norm = minmax_normalize(capped)
    kept = [(cid, norm[cid]) for cid, _ in ranked if norm[cid] >= discard_below]
    return kept
