from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import selfies as sf
import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            self.norm_q(query),
            self.norm_kv(context),
            self.norm_kv(context),
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        hidden = query + self.dropout(attended)
        return hidden + self.dropout(self.ff(self.norm_ff(hidden)))


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, width: int = 320, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.p2d = nn.ModuleList([CrossAttentionBlock(width, heads) for _ in range(layers)])
        self.d2p = nn.ModuleList([CrossAttentionBlock(width, heads) for _ in range(layers)])

    def forward(
        self,
        protein: torch.Tensor,
        drug: torch.Tensor,
        protein_padding_mask: torch.Tensor,
        drug_padding_mask: torch.Tensor,
        protein_interface_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if protein_interface_mask is not None:
            allowed = protein_interface_mask.bool() & ~protein_padding_mask
            if not bool(allowed.any(dim=1).all()):
                raise ValueError("Every pair must have an attended protein interface")
            protein_padding_mask = ~allowed
        for protein_block, drug_block in zip(self.p2d, self.d2p, strict=True):
            next_protein = protein_block(protein, drug, drug_padding_mask)
            if protein_interface_mask is not None:
                next_protein = torch.where(allowed[:, :, None], next_protein, protein)
            next_drug = drug_block(drug, protein, protein_padding_mask)
            protein, drug = next_protein, next_drug
        return protein, drug


class LogiCAModel(nn.Module):
    """Protein-ligand LogiCA architecture reconstructed from the released state dict.

    The public checkpoint contains only weights that were trainable during
    BindingDB pretraining. Base encoder weights are loaded from their original
    Hugging Face repositories, matching the authors' model card.
    """

    def __init__(
        self,
        protein_model: str,
        drug_model: str,
        width: int = 320,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.esm = AutoModelForMaskedLM.from_pretrained(protein_model, cache_dir=cache_dir)
        self.drug_encoder = AutoModelForMaskedLM.from_pretrained(drug_model, cache_dir=cache_dir)
        protein_width = int(self.esm.config.hidden_size)
        drug_width = int(self.drug_encoder.config.hidden_size)
        self.prot_gate = nn.Parameter(torch.tensor([-6.0]))
        self.drug_gate = nn.Parameter(torch.tensor([-6.0]))
        self.prot_proj_in = nn.Linear(protein_width, width, bias=False)
        self.prot_proj_out = nn.Linear(width, protein_width, bias=False)
        self.drug_proj_in = nn.Linear(drug_width, width, bias=False)
        self.drug_proj_out = nn.Linear(width, drug_width, bias=False)
        for projection in (
            self.prot_proj_in,
            self.prot_proj_out,
            self.drug_proj_in,
            self.drug_proj_out,
        ):
            self._init_projection(projection)
        self.cross_attn = BidirectionalCrossAttention(width=width, heads=4, layers=2)
        # Present in the official downstream implementation but absent from the
        # trainable-only pretraining checkpoint.  Zero therefore reproduces the
        # official symmetric initialization when that checkpoint is loaded.
        self.pair_alpha_logit = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _init_projection(layer: nn.Linear) -> None:
        """Official partial-identity initialization for modality projections."""
        with torch.no_grad():
            layer.weight.zero_()
            diagonal = min(layer.out_features, layer.in_features)
            layer.weight[:diagonal, :diagonal] = torch.eye(diagonal)

    def pair_alpha(self) -> torch.Tensor:
        """Learned protein-side weight used by official binding scoring."""
        return torch.sigmoid(self.pair_alpha_logit)

    def encode_hidden(
        self,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        drug_input_ids: torch.Tensor,
        drug_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        protein_output = self.esm.base_model(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            return_dict=True,
        )
        drug_output = self.drug_encoder.base_model(
            input_ids=drug_input_ids,
            attention_mask=drug_attention_mask,
            return_dict=True,
        )
        return protein_output.last_hidden_state, drug_output.last_hidden_state

    def forward_from_hidden(
        self,
        protein_hidden: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        drug_hidden: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        protein_interface_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        protein_interaction_0 = self.prot_proj_in(protein_hidden)
        drug_interaction_0 = self.drug_proj_in(drug_hidden)
        protein_interaction, drug_interaction = self.cross_attn(
            protein_interaction_0,
            drug_interaction_0,
            protein_attention_mask == 0,
            drug_attention_mask == 0,
            protein_interface_mask=protein_interface_mask,
        )
        protein_contextual = protein_hidden + torch.sigmoid(self.prot_gate) * self.prot_proj_out(
            protein_interaction - protein_interaction_0
        )
        drug_contextual = drug_hidden + torch.sigmoid(self.drug_gate) * self.drug_proj_out(
            drug_interaction - drug_interaction_0
        )
        protein_logits = self.esm.lm_head(protein_contextual)
        drug_logits = self.drug_encoder.lm_head(drug_contextual)
        protein_unconditional = self.esm.lm_head(protein_hidden)
        drug_unconditional = self.drug_encoder.lm_head(drug_hidden)
        return protein_logits, drug_logits, protein_unconditional, drug_unconditional

    def forward(
        self,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        drug_input_ids: torch.Tensor,
        drug_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        protein_hidden, drug_hidden = self.encode_hidden(
            protein_input_ids,
            protein_attention_mask,
            drug_input_ids,
            drug_attention_mask,
        )
        return self.forward_from_hidden(
            protein_hidden,
            protein_attention_mask,
            drug_hidden,
            drug_attention_mask,
        )


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _make_masked_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    mask_token_id: int,
    keys: list[str],
    mask_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = input_ids.clone()
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, key in enumerate(keys):
        candidates = torch.nonzero(
            (attention_mask[row] == 1) & (special_tokens_mask[row] == 0),
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"No scoreable tokens for {key}")
        count = max(1, int(round(candidates.numel() * mask_fraction)))
        generator = torch.Generator().manual_seed(_stable_seed(key, seed))
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
        selected[row, chosen] = True
        masked[row, chosen] = mask_token_id
    return masked, selected


def _make_positional_masked_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    mask_token_id: int,
    residue_indices: Sequence[int],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask an explicit, prespecified set of zero-based residue positions.

    The ESM-2 tokenizer emits one leading <cls> token, so residue ``i`` occupies
    token ``i + 1``. This bypasses ``mask_fraction`` entirely: the caller states
    exactly which residues define the score, which is what a structure-derived
    pocket region requires.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("Positional protein masking expects a single protein row")
    ordered = sorted(set(int(index) for index in residue_indices))
    if not ordered:
        raise ValueError("At least one residue index is required")
    if ordered[0] < 0 or ordered[-1] >= sequence_length:
        raise ValueError(
            f"Residue indices must lie in [0, {sequence_length - 1}]; "
            f"got min={ordered[0]}, max={ordered[-1]}"
        )
    token_positions = torch.tensor([index + 1 for index in ordered], dtype=torch.long)
    if int(token_positions.max()) >= input_ids.shape[1]:
        raise ValueError("Residue index exceeds the tokenized protein length; check truncation")
    if not bool(attention_mask[0, token_positions].all()):
        raise ValueError("A requested residue maps to an unattended token position")
    if bool(special_tokens_mask[0, token_positions].any()):
        raise ValueError("A requested residue maps to a special token position")
    masked = input_ids.clone()
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    selected[0, token_positions] = True
    masked[0, token_positions] = mask_token_id
    return masked, selected


def _make_token_likelihood_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    residue_indices: Sequence[int] | None = None,
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the sequence's own tokens with no masking at all.

    This reproduces the reference implementation's ``pair_scores_from_tokens``,
    the score function the official code selects for ``method == "logica"`` on
    the binding task: inputs stay unmasked and every valid position (not
    padding, not a special token) contributes its own-token log-probability.

    ``residue_indices`` optionally narrows the scored set to a structure-derived
    region, which is the paper's evaluation-position set A restricted to
    binding-interface residues. Unlike the masked path this only changes which
    positions are *read*, never what the encoder sees.
    """
    selected = attention_mask.bool() & ~special_tokens_mask.bool()
    if residue_indices is not None:
        if sequence_length is None:
            raise ValueError("sequence_length is required to map residue indices")
        if input_ids.shape[0] != 1:
            raise ValueError("Region-restricted scoring expects a single protein row")
        ordered = sorted(set(int(index) for index in residue_indices))
        if not ordered:
            raise ValueError("At least one residue index is required")
        if ordered[0] < 0 or ordered[-1] >= sequence_length:
            raise ValueError(
                f"Residue indices must lie in [0, {sequence_length - 1}]; "
                f"got min={ordered[0]}, max={ordered[-1]}"
            )
        region = torch.zeros_like(selected)
        token_positions = torch.tensor([index + 1 for index in ordered], dtype=torch.long)
        if int(token_positions.max()) >= input_ids.shape[1]:
            raise ValueError("Residue index exceeds the tokenized protein length")
        region[0, token_positions] = True
        selected = selected & region
    if not bool(selected.any()):
        raise ValueError("No scoreable positions remain after restriction")
    return input_ids.clone(), selected


def _mean_log_probability(
    logits: torch.Tensor,
    original_ids: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, original_ids.unsqueeze(-1)).squeeze(-1)
    selected_float = selected.to(token_log_probs.dtype)
    return (token_log_probs * selected_float).sum(dim=1) / selected_float.sum(dim=1).clamp_min(1)


@dataclass
class PreparedPairs:
    smiles: tuple[str, ...]
    protein_hidden: torch.Tensor
    protein_attention_mask: torch.Tensor
    protein_original_ids: torch.Tensor
    protein_selected: torch.Tensor
    drug_hidden: torch.Tensor
    drug_attention_mask: torch.Tensor
    drug_original_ids: torch.Tensor
    drug_selected: torch.Tensor

    def __len__(self) -> int:
        return len(self.smiles)


class LogiCABindingScorer:
    def __init__(
        self,
        checkpoint: Path,
        protein_model: str = "facebook/esm2_t6_8M_UR50D",
        drug_model: str = "HUBioDataLab/SELFormer",
        device: str = "cpu",
        mask_fraction: float = 0.15,
        alpha: float = 0.5,
        seed: int = 2026,
        hf_cache: Path | None = None,
        protein_mask_key: str | None = None,
        protein_mask_positions: Sequence[int] | None = None,
        protein_mask_region: str = "random",
        score_mode: str = "masked_gain",
    ) -> None:
        if score_mode not in ("masked_gain", "token_likelihood"):
            raise ValueError(
                "score_mode must be 'masked_gain' (this project's original masked "
                "context-gain score) or 'token_likelihood' (the reference "
                "pair_scores_from_tokens score used for binding)"
            )
        if not 0 < mask_fraction <= 1:
            raise ValueError("mask_fraction must be in (0, 1]")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        cache_dir = str(hf_cache) if hf_cache else None
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model, cache_dir=cache_dir)
        self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_model, cache_dir=cache_dir)
        self.model = LogiCAModel(protein_model, drug_model, cache_dir=cache_dir)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        incompatible = self.model.load_state_dict(state, strict=False)
        critical_prefixes = (
            "prot_gate",
            "drug_gate",
            "prot_proj_",
            "drug_proj_",
            "cross_attn.",
        )
        critical_missing = [
            key for key in incompatible.missing_keys if key.startswith(critical_prefixes)
        ]
        if critical_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: critical_missing={critical_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.checkpoint_epoch = payload.get("epoch")
        self.checkpoint_metrics = payload.get("metrics", {})
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mask_fraction = mask_fraction
        self.alpha = alpha
        if score_mode == "token_likelihood":
            with torch.no_grad():
                clipped = min(max(float(alpha), 1e-6), 1 - 1e-6)
                self.model.pair_alpha_logit.fill_(float(np.log(clipped / (1 - clipped))))
        self.seed = seed
        self.protein_mask_key = protein_mask_key
        self.protein_mask_positions = (
            tuple(sorted(set(int(index) for index in protein_mask_positions)))
            if protein_mask_positions is not None
            else None
        )
        self.protein_mask_region = protein_mask_region
        self.score_mode = score_mode

    @staticmethod
    def smiles_to_selfies(smiles: str) -> str:
        try:
            return sf.encoder(smiles)
        except sf.EncoderError as exc:
            raise ValueError(f"SELFIES conversion failed for {smiles!r}") from exc

    @torch.no_grad()
    def prepare(
        self,
        protein_sequence: str,
        smiles: Iterable[str],
        encoder_batch_size: int = 8,
    ) -> tuple[PreparedPairs, dict[str, int | float | str]]:
        compounds = list(smiles)
        if not compounds:
            raise ValueError("At least one compound is required")
        selfies_batch = [self.smiles_to_selfies(value) for value in compounds]
        protein_tokens = self.protein_tokenizer(
            [protein_sequence],
            padding=True,
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        drug_tokens = self.drug_tokenizer(
            selfies_batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        protein_original = protein_tokens["input_ids"]
        drug_original = drug_tokens["input_ids"]
        if self.score_mode == "token_likelihood":
            protein_masked, protein_selected = _make_token_likelihood_inputs(
                protein_original,
                protein_tokens["attention_mask"],
                protein_tokens["special_tokens_mask"],
                self.protein_mask_positions,
                len(protein_sequence),
            )
            drug_masked, drug_selected = _make_token_likelihood_inputs(
                drug_original,
                drug_tokens["attention_mask"],
                drug_tokens["special_tokens_mask"],
            )
        elif self.protein_mask_positions is None:
            protein_masked, protein_selected = _make_masked_inputs(
                protein_original,
                protein_tokens["attention_mask"],
                protein_tokens["special_tokens_mask"],
                int(self.protein_tokenizer.mask_token_id),
                [self.protein_mask_key or protein_sequence],
                self.mask_fraction,
                self.seed,
            )
            drug_masked, drug_selected = _make_masked_inputs(
                drug_original,
                drug_tokens["attention_mask"],
                drug_tokens["special_tokens_mask"],
                int(self.drug_tokenizer.mask_token_id),
                compounds,
                self.mask_fraction,
                self.seed,
            )
        else:
            protein_masked, protein_selected = _make_positional_masked_inputs(
                protein_original,
                protein_tokens["attention_mask"],
                protein_tokens["special_tokens_mask"],
                int(self.protein_tokenizer.mask_token_id),
                self.protein_mask_positions,
                len(protein_sequence),
            )
            drug_masked, drug_selected = _make_masked_inputs(
                drug_original,
                drug_tokens["attention_mask"],
                drug_tokens["special_tokens_mask"],
                int(self.drug_tokenizer.mask_token_id),
                compounds,
                self.mask_fraction,
                self.seed,
            )
        protein_hidden = self.model.esm.base_model(
            input_ids=protein_masked.to(self.device),
            attention_mask=protein_tokens["attention_mask"].to(self.device),
            return_dict=True,
        ).last_hidden_state.cpu()
        drug_hidden_parts: list[torch.Tensor] = []
        for offset in range(0, len(compounds), encoder_batch_size):
            end = offset + encoder_batch_size
            hidden = self.model.drug_encoder.base_model(
                input_ids=drug_masked[offset:end].to(self.device),
                attention_mask=drug_tokens["attention_mask"][offset:end].to(self.device),
                return_dict=True,
            ).last_hidden_state
            drug_hidden_parts.append(hidden.cpu())
        prepared = PreparedPairs(
            smiles=tuple(compounds),
            protein_hidden=protein_hidden,
            protein_attention_mask=protein_tokens["attention_mask"],
            protein_original_ids=protein_original,
            protein_selected=protein_selected,
            drug_hidden=torch.cat(drug_hidden_parts, dim=0),
            drug_attention_mask=drug_tokens["attention_mask"],
            drug_original_ids=drug_original,
            drug_selected=drug_selected,
        )
        diagnostics: dict[str, int | float | str] = {
            "checkpoint_epoch": int(self.checkpoint_epoch or -1),
            "protein_gate": float(torch.sigmoid(self.model.prot_gate).item()),
            "drug_gate": float(torch.sigmoid(self.model.drug_gate).item()),
            "max_protein_tokens": int(protein_tokens["attention_mask"].sum(1).max()),
            "max_drug_tokens": int(drug_tokens["attention_mask"].sum(1).max()),
            "mask_fraction": self.mask_fraction,
            "alpha": float(
                self.model.pair_alpha().detach().cpu()
                if self.score_mode == "token_likelihood"
                else self.alpha
            ),
            "protein_mask_region": self.protein_mask_region,
            "score_mode": self.score_mode,
            "protein_scored_positions": int(protein_selected.sum()),
            "drug_scored_positions": int(drug_selected.sum(dim=1).max()),
        }
        return prepared, diagnostics

    def score_tensor(self, prepared: PreparedPairs, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(dtype=torch.long, device="cpu")
        batch_size = int(indices.numel())
        protein_hidden = prepared.protein_hidden.expand(batch_size, -1, -1).to(self.device)
        protein_attention = prepared.protein_attention_mask.expand(batch_size, -1).to(self.device)
        protein_original = prepared.protein_original_ids.expand(batch_size, -1).to(self.device)
        protein_selected = prepared.protein_selected.expand(batch_size, -1).to(self.device)
        drug_hidden = prepared.drug_hidden[indices].to(self.device)
        drug_attention = prepared.drug_attention_mask[indices].to(self.device)
        drug_original = prepared.drug_original_ids[indices].to(self.device)
        drug_selected = prepared.drug_selected[indices].to(self.device)
        protein_logits, drug_logits, protein_unconditional, drug_unconditional = (
            self.model.forward_from_hidden(
                protein_hidden,
                protein_attention,
                drug_hidden,
                drug_attention,
            )
        )
        protein_conditional = _mean_log_probability(
            protein_logits, protein_original, protein_selected
        )
        drug_conditional = _mean_log_probability(drug_logits, drug_original, drug_selected)
        if self.score_mode == "token_likelihood":
            # Reference pair_scores_from_tokens: conditional log-likelihood only,
            # with no unconditional baseline term.
            alpha = self.model.pair_alpha()
            return alpha * protein_conditional + (1 - alpha) * drug_conditional
        protein_context_gain = protein_conditional - _mean_log_probability(
            protein_unconditional, protein_original, protein_selected
        )
        drug_context_gain = drug_conditional - _mean_log_probability(
            drug_unconditional, drug_original, drug_selected
        )
        return self.alpha * protein_context_gain + (1 - self.alpha) * drug_context_gain

    @torch.inference_mode()
    def score_prepared(self, prepared: PreparedPairs, batch_size: int = 1) -> np.ndarray:
        self.model.eval()
        values: list[float] = []
        for offset in range(0, len(prepared), batch_size):
            indices = torch.arange(offset, min(offset + batch_size, len(prepared)))
            values.extend(float(value) for value in self.score_tensor(prepared, indices).cpu())
        return np.asarray(values, dtype=np.float64)

    def configure_finetuning(self) -> list[nn.Parameter]:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        modules: tuple[nn.Module, ...] = (
            self.model.prot_proj_in,
            self.model.prot_proj_out,
            self.model.drug_proj_in,
            self.model.drug_proj_out,
            self.model.cross_attn,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.model.prot_gate.requires_grad_(True)
        self.model.drug_gate.requires_grad_(True)
        if self.score_mode == "token_likelihood":
            self.model.pair_alpha_logit.requires_grad_(True)
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def score(
        self,
        protein_sequence: str,
        smiles: Iterable[str],
        batch_size: int = 1,
    ) -> tuple[np.ndarray, dict[str, int | float | str]]:
        prepared, diagnostics = self.prepare(
            protein_sequence,
            smiles,
            encoder_batch_size=max(1, batch_size),
        )
        return self.score_prepared(prepared, batch_size=batch_size), diagnostics
