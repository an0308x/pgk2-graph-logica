from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .graph_model import (
    LIGAND_EDGE_DIM,
    LIGAND_FEATURE_DIM,
    POCKET_EDGE_DIM,
    POCKET_FEATURE_DIM,
    GraphEncoder,
    PocketGraph,
)


@dataclass(frozen=True)
class PackedLigandBatch:
    """A disconnected batch of ligand graphs with local graphs concatenated."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_offsets: torch.Tensor
    fingerprints: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.node_offsets.numel() - 1)

    def to(self, device: torch.device | str) -> "PackedLigandBatch":
        return replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_offsets=self.node_offsets.to(device),
            fingerprints=(
                None if self.fingerprints is None else self.fingerprints.to(device)
            ),
        )


def _dense_ligand_nodes(
    hidden: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        hidden[int(offsets[row]) : int(offsets[row + 1])]
        for row in range(offsets.numel() - 1)
    ]
    if not sequences or any(sequence.shape[0] == 0 for sequence in sequences):
        raise ValueError("Every packed ligand graph must contain at least one atom")
    dense = pad_sequence(sequences, batch_first=True)
    lengths = torch.as_tensor(
        [sequence.shape[0] for sequence in sequences], device=hidden.device
    )
    padding = torch.arange(dense.shape[1], device=hidden.device)[None, :] >= lengths[:, None]
    return dense, padding


def _masked_mean(hidden: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    present = (~padding).to(hidden.dtype)
    return (hidden * present[:, :, None]).sum(dim=1) / present.sum(dim=1).clamp_min(1)[:, None]


def _masked_max(hidden: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    return hidden.masked_fill(padding[:, :, None], -torch.inf).amax(dim=1)


class PGK2PocketMultitaskModel(nn.Module):
    """Scalable ligand GNN with a fixed, ESM-contextualized PGK2 pocket.

    Ligand atoms may attend only to the structure-defined direct-pocket nodes.
    The complete PGK2 sequence is encoded before the selected residue embeddings
    are injected into each fixed structure template.  No ligand--protein
    distances are fabricated because ligand coordinates are not protein-aligned.
    """

    def __init__(
        self,
        esm_width: int,
        output_dim: int,
        graph_width: int = 128,
        heads: int = 4,
        mode: str = "pocket",
    ) -> None:
        super().__init__()
        if mode not in {"pocket", "ligand"}:
            raise ValueError("mode must be 'pocket' or 'ligand'")
        self.mode = mode
        self.ligand_encoder = GraphEncoder(
            LIGAND_FEATURE_DIM, LIGAND_EDGE_DIM, graph_width, layers=3
        )
        self.pocket_encoder = GraphEncoder(
            POCKET_FEATURE_DIM, POCKET_EDGE_DIM, graph_width, layers=2
        )
        self.esm_projection = nn.Linear(esm_width, graph_width, bias=False)
        self.pocket_context_norm = nn.LayerNorm(graph_width)
        self.ligand_cross_attention = nn.MultiheadAttention(
            graph_width, heads, batch_first=True
        )
        self.pocket_cross_attention = nn.MultiheadAttention(
            graph_width, heads, batch_first=True
        )
        self.ligand_cross_norm = nn.LayerNorm(graph_width)
        self.pocket_cross_norm = nn.LayerNorm(graph_width)
        readout_width = graph_width * (3 if mode == "pocket" else 2)
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_width),
            nn.Linear(readout_width, graph_width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(graph_width * 2, output_dim),
        )

    def _template_readout(
        self,
        ligand_dense: torch.Tensor,
        ligand_padding: torch.Tensor,
        pocket: PocketGraph,
        full_sequence_esm: torch.Tensor,
    ) -> torch.Tensor:
        pocket_hidden = self.pocket_encoder(
            pocket.node_features, pocket.edge_index, pocket.edge_features
        )
        pocket_hidden = self.pocket_context_norm(
            pocket_hidden
            + self.esm_projection(full_sequence_esm[pocket.residue_indices])
        )
        direct = pocket_hidden[pocket.direct_mask]
        direct_batch = direct.unsqueeze(0).expand(ligand_dense.shape[0], -1, -1)
        ligand_update, _ = self.ligand_cross_attention(
            ligand_dense, direct_batch, direct_batch, need_weights=False
        )
        ligand_context = self.ligand_cross_norm(ligand_dense + ligand_update)
        pocket_update, _ = self.pocket_cross_attention(
            direct_batch,
            ligand_dense,
            ligand_dense,
            key_padding_mask=ligand_padding,
            need_weights=False,
        )
        pocket_context = self.pocket_cross_norm(direct_batch + pocket_update)
        return torch.cat(
            [
                _masked_mean(ligand_context, ligand_padding),
                _masked_max(ligand_context, ligand_padding),
                pocket_context.mean(dim=1),
            ],
            dim=-1,
        )

    def forward(
        self,
        batch: PackedLigandBatch,
        pocket_graphs: Sequence[PocketGraph],
        full_sequence_esm: torch.Tensor,
    ) -> torch.Tensor:
        ligand_hidden = self.ligand_encoder(
            batch.node_features, batch.edge_index, batch.edge_features
        )
        ligand_dense, ligand_padding = _dense_ligand_nodes(
            ligand_hidden, batch.node_offsets
        )
        if self.mode == "ligand":
            representation = torch.cat(
                [
                    _masked_mean(ligand_dense, ligand_padding),
                    _masked_max(ligand_dense, ligand_padding),
                ],
                dim=-1,
            )
        else:
            if not pocket_graphs:
                raise ValueError("Pocket-conditioned mode requires structure templates")
            if full_sequence_esm.ndim != 2:
                raise ValueError("full_sequence_esm must have shape (residues, width)")
            template_representations = [
                self._template_readout(
                    ligand_dense, ligand_padding, pocket, full_sequence_esm
                )
                for pocket in pocket_graphs
            ]
            representation = torch.stack(template_representations).mean(dim=0)
        return self.readout(representation)


class FingerprintMultitaskModel(nn.Module):
    def __init__(self, fingerprint_bits: int, output_dim: int, width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(fingerprint_bits, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, output_dim),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.network(fingerprints)
