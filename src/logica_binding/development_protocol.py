"""Prespecified DEL sensitivity experiments, not calibrated inhibition labels."""
from __future__ import annotations

import numpy as np
import torch
from .streaming_graph_logica import CHANNEL_WEIGHTS

PROTOCOL = "pgk2_dev_v4_zero_intervals_objective_ablation"


def evidence_intervals(frame, zero_cap=0):
    """Keep v3 observed evidence; optionally add weak, bounded zero-I evidence.

    The zero cap is an assumed count range, NOT a confidence interval. Only
    competition zeros with mean target >=5 are added, with <=0.05 confidence,
    downweighted by historic hits and observed background. NTC zeros remain
    excluded. Original metadata/labels are never rewritten.
    """
    if zero_cap not in (0, 1, 5):
        raise ValueError("Prespecified zero caps are 0 (exclude), 1, and 5")
    lower = frame.select([f"proxy_{i}" for i in range(3)]).to_numpy().astype("float32").copy()
    upper = lower.copy()
    confidence = frame.select([f"confidence_{i}" for i in range(3)]).to_numpy().astype("float32").copy()
    if not zero_cap:
        return lower, upper, confidence
    def col(name):
        x = np.asarray(frame[name], dtype=np.float64)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Invalid counts: {name}")
        return x
    n = np.maximum(col("source_rows"), 1)
    target = col("count_PGK2") / n
    inhibitor = col("count_PGK2_with_inhibitor") / n
    background = np.maximum(col("count_NTC_selection") / n,
        col("count_NTC_supplement") / np.maximum(col("ntc_supplement_rows"), 1))
    added = (target >= 5) & (inhibitor == 0)
    lower[added, 0] = np.log1p(target[added]) - np.log1p(zero_cap)
    upper[added, 0] = np.log1p(target[added])
    confidence[added, 0] = (.05 * target[added] / (target[added] + 20)
        / (1 + col("historic_hits")[added])
        / (1 + background[added] / target[added]))
    return lower, upper, confidence


def interval_pair_loss(scores, lower, upper, confidence, temperature=.1, margin=np.log(2)):
    """Rank only when assumed proxy intervals are separated by the margin."""
    if temperature <= 0 or margin <= 0 or torch.any(lower > upper):
        raise ValueError("Invalid interval ranking configuration")
    total, stats = scores.sum() * 0, []
    triangle = torch.triu(torch.ones((len(scores), len(scores)), device=scores.device,
                                    dtype=torch.bool), diagonal=1)
    for i, channel_weight in enumerate(CHANNEL_WEIGHTS):
        positive = lower[:, i, None] - upper[None, :, i] >= margin
        negative = lower[None, :, i] - upper[:, i, None] >= margin
        valid = triangle & (positive | negative)
        valid &= (confidence[:, i, None] > 0) & (confidence[None, :, i] > 0)
        a, b = valid.nonzero(as_tuple=True)
        direction = torch.where(positive[a, b], 1., -1.)
        weight = torch.minimum(confidence[a, i], confidence[b, i])
        predicted = (scores[a] - scores[b]) * direction
        numerator = (weight * torch.nn.functional.softplus(-predicted / temperature)).sum()
        total = total + channel_weight * numerator / max(1, len(a))
        wins = (predicted > 0).to(weight.dtype) + .5 * (predicted == 0).to(weight.dtype)
        stats.append({"pairs": len(a), "loss_sum": float(numerator.detach()),
            "weight_sum": float(weight.sum().detach()), "weighted_wins": float((weight*wins).sum().detach())})
    return total / sum(CHANNEL_WEIGHTS), stats


def training_batches(frame, confidence, seed, batch_size=32):
    """Visit every eligible train molecule per channel; never draw dev/holdout."""
    if batch_size < 2:
        raise ValueError("Ranking requires at least two molecules")
    rng = np.random.default_rng(seed)
    train = frame["split"].to_numpy() == "train"
    for channel in range(3):
        pool = rng.permutation(np.flatnonzero(train & (confidence[:, channel] > 0)))
        # A singleton tail joins the previous batch rather than being dropped.
        for offset in range(0, len(pool), batch_size):
            if offset and len(pool) - offset == 1:
                break
            stop = min(offset+batch_size, len(pool))
            if len(pool) - stop == 1:
                stop += 1
            if stop-offset >= 2:
                yield pool[offset:stop]


def select_checkpoint(metric, best, stale):
    """Shared epoch-zero/intermediate selection; ties never replace earlier state."""
    improved = metric is not None and np.isfinite(metric) and metric < best
    return (float(metric), 0, True) if improved else (best, stale+1, False)


def gradient_comparison(rank_loss, aux_loss, parameters):
    """Scaled objective gradient norms/cosine, including shared parameters only."""
    r = torch.autograd.grad(rank_loss, parameters, retain_graph=True, allow_unused=True)
    a = torch.autograd.grad(aux_loss, parameters, retain_graph=True, allow_unused=True)
    def norm(values):
        return float(torch.sqrt(sum((x.detach().float().square().sum() for x in values if x is not None),
                                    rank_loss.new_zeros(()))))
    shared = [(x,y) for x,y in zip(r,a) if x is not None and y is not None]
    rn, an = norm([x for x,_ in shared]), norm([y for _,y in shared])
    dot = sum((float((x.detach().float()*y.detach().float()).sum()) for x,y in shared), 0.)
    return {"ranking_norm": norm(r), "auxiliary_norm": norm(a), "shared_ranking_norm": rn,
        "shared_auxiliary_norm": an, "shared_cosine": dot/(rn*an) if rn*an else None}
