"""Packed Graph LogiCA, with a separate ligand-only auxiliary objective.

The binding score is still the native LogiCA token log-likelihood, not a new
regression head. Molecular reconstruction never sees the protein and never
declares the unlabeled library to be PGK2 binders.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .full_graph_model import PackedLigandBatch, _dense_ligand_nodes
from .graph_model import GraphLogiCAModel, PocketGraph
from .model import _mean_log_probability

FULL_PROTOCOL = "pgk2_full_logica_v3_observed_controls_ligand_aux"
CHANNELS = ("competition", "selection_ntc", "supplement_ntc")
CHANNEL_WEIGHTS = (2.0, 1.0, 1.0)


def observed_evidence(frame):
    """Relative DEL preferences only where both measurements were observed.

    Duplicate source rows are averaged rather than treated as independent
    replicates. No normalized Z column is read. Counts across unknown depths
    do not define calibrated enrichment or inhibition probabilities.
    """
    def col(name):
        result = np.asarray(frame[name], dtype=np.float64)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError(f"Invalid observation in {name}")
        return result
    n = np.maximum(col("source_rows"), 1)
    target = col("count_PGK2") / n
    controls = np.column_stack([
        col("count_PGK2_with_inhibitor") / n,
        col("count_NTC_selection") / n,
        col("count_NTC_supplement") / np.maximum(col("ntc_supplement_rows"), 1),
    ])
    observed = (target[:, None] > 0) & (controls > 0)
    proxy = np.log1p(target[:, None]) - np.log1p(controls)
    confidence = (target[:, None] / (target[:, None] + 20)
                  * controls / (controls + 5)
                  / (1 + col("historic_hits")[:, None]))
    return np.where(observed, proxy, 0).astype("float32"), np.where(
        observed, confidence, 0).astype("float32")


def evidence_pair_loss(scores, proxy, confidence, temperature=0.1, margin=np.log(2)):
    """Confidence-scaled ranking; zero/missing controls cannot form pairs.

    Absolute confidence is preserved (denominator is pair count, not sum of
    weights). Each unordered pair is considered once per channel. No label
    derives from the model's own predictions.
    """
    if temperature <= 0 or margin <= 0:
        raise ValueError("Positive temperature and margin required")
    total = scores.sum() * 0
    stats = []
    upper = torch.triu(torch.ones((len(scores), len(scores)), dtype=torch.bool,
                                 device=scores.device), diagonal=1)
    for channel, channel_weight in enumerate(CHANNEL_WEIGHTS):
        delta = proxy[:, channel, None] - proxy[None, :, channel]
        valid = upper & (delta.abs() >= margin)
        valid &= (confidence[:, channel, None] > 0) & (confidence[None, :, channel] > 0)
        a, b = valid.nonzero(as_tuple=True)
        weight = torch.minimum(confidence[a, channel], confidence[b, channel])
        direction = delta[a, b].sign()
        predicted = (scores[a] - scores[b]) * direction
        losses = nn.functional.softplus(-predicted / temperature)
        numerator = (weight * losses).sum()
        total = total + channel_weight * numerator / max(1, len(a))
        wins = (predicted > 0).to(weight.dtype) + .5 * (predicted == 0).to(weight.dtype)
        stats.append({"pairs": len(a), "loss_sum": float(numerator.detach()),
                      "weight_sum": float(weight.sum().detach()),
                      "weighted_wins": float((weight * wins).sum().detach())})
    return total / sum(CHANNEL_WEIGHTS), stats


def encode_packed(model: GraphLogiCAModel, batch: PackedLigandBatch):
    """Same ligand GNN as the row-wise model, with disconnected batching."""
    lengths = batch.node_offsets[1:] - batch.node_offsets[:-1]
    owner = torch.repeat_interleave(torch.arange(batch.batch_size, device=lengths.device), lengths)
    encoder = model.conditioner.ligand_encoder
    hidden = encoder.input(batch.node_features)
    has_edges = torch.zeros(batch.batch_size, dtype=torch.bool, device=hidden.device)
    if batch.edge_index.numel():
        has_edges[owner[batch.edge_index[1]]] = True
    for layer in encoder.layers:
        update = layer(hidden, batch.edge_index, batch.edge_features)
        # A whole single-atom graph with no edges is a no-op in the reference.
        hidden = torch.where(has_edges[owner, None], update, hidden)
    return hidden, owner


def ligand_condition(model, drug_hidden, ligand_hidden, offsets):
    dense, padding = _dense_ligand_nodes(ligand_hidden, offsets)
    c = model.conditioner
    attended, _ = c.drug_attention(c.drug_query(drug_hidden), dense, dense,
                                   key_padding_mask=padding, need_weights=False)
    return drug_hidden + torch.sigmoid(c.drug_graph_gate) * c.drug_out(attended)


def ligand_reconstruction_loss(model, masked_hidden, original_ids, selected,
                               batch: PackedLigandBatch):
    """Ligand-only graph-to-token reconstruction; no PGK2 inputs or gradients."""
    nodes, _ = encode_packed(model, batch)
    hidden = ligand_condition(model, masked_hidden, nodes, batch.node_offsets)
    logits = model.base.drug_encoder.lm_head(hidden)
    return -_mean_log_probability(logits.float(), original_ids, selected).mean()


def packed_logica_scores(model: GraphLogiCAModel, protein_hidden, drug_hidden,
                         drug_attention, drug_ids, drug_selected,
                         protein_ids, pockets: list[PocketGraph], batch: PackedLigandBatch):
    """Batched counterpart of score_graph_prepared, exactly the same score.

    After full-sequence ESM encoding and graph conditioning, only direct-pocket
    tokens need enter base cross-attention. Masked-out tokens cannot affect it.
    """
    if not pockets:
        raise ValueError("A pocket template is required")
    c = model.conditioner
    ligand_base, owner = encode_packed(model, batch)
    b = batch.batch_size
    results = []
    for pocket in pockets:
        tokens = pocket.residue_indices + 1
        direct = pocket.direct_mask.nonzero(as_tuple=True)[0]
        p = protein_hidden[:, tokens].expand(b, -1, -1)
        if model.graph_conditioning:
            pocket_base = c.pocket_encoder(pocket.node_features, pocket.edge_index, pocket.edge_features)
            pocket_nodes = c.protein_context_norm(pocket_base[None] + c.protein_context_in(p))
            # Every atom connects to every direct-pocket node of its own row.
            atom_index = torch.arange(len(ligand_base), device=owner.device).repeat_interleave(len(direct))
            pocket_local = direct.repeat(len(ligand_base))
            pocket_index = owner[atom_index] * len(tokens) + pocket_local
            flat_pocket = pocket_nodes.reshape(-1, pocket_nodes.shape[-1])
            features = ligand_base.new_zeros((len(atom_index), 5)); features[:, -1] = 1
            to_p = c.ligand_to_pocket(torch.cat([ligand_base[atom_index], flat_pocket[pocket_index], features], -1))
            to_l = c.pocket_to_ligand(torch.cat([flat_pocket[pocket_index], ligand_base[atom_index], features], -1))
            p_aggregate = torch.zeros_like(flat_pocket).index_add(0, pocket_index, to_p)
            l_aggregate = torch.zeros_like(ligand_base).index_add(0, atom_index, to_l)
            pocket_nodes = c.pocket_norm(flat_pocket + p_aggregate).reshape_as(pocket_nodes)
            ligand_nodes = c.ligand_norm(ligand_base + l_aggregate)
            p = p + torch.sigmoid(c.protein_graph_gate) * c.protein_out(pocket_nodes)
            d = ligand_condition(model, drug_hidden, ligand_nodes, batch.node_offsets)
        else:
            d = drug_hidden
        p = p[:, direct]
        p_logits, d_logits, _, _ = model.base.forward_from_hidden(
            p, torch.ones(p.shape[:2], device=p.device), d, drug_attention)
        p_ids = protein_ids[:, tokens[direct]].expand(b, -1)
        ps = _mean_log_probability(p_logits.float(), p_ids, torch.ones_like(p_ids, dtype=torch.bool))
        ds = _mean_log_probability(d_logits.float(), drug_ids, drug_selected)
        alpha = model.base.pair_alpha()
        results.append(alpha * ps + (1-alpha) * ds)
    return torch.stack(results).mean(0)
